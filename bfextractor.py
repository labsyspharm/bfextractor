import math
import warnings
import sys
import os
import io
import itertools
import pathlib
import uuid
import json
from xml.etree import ElementTree
import boto3
import s3fs
import s3transfer.manager
import s3transfer.subscribers
import numpy as np
import skimage.io
import skimage.transform
import jnius
import logging, time
import tifffile
import zarr

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s - %(message)s')
logger = logging.getLogger("minerva")
logger.setLevel(logging.INFO)

OME_NS = 'http://www.openmicroscopy.org/Schemas/OME/2016-06'

def tile(img, tile_size):
    for yi, y in enumerate(range(0, img.shape[0], tile_size)):
        for xi, x in enumerate(range(0, img.shape[1], tile_size)):
            yield yi, xi, img[y:y+tile_size, x:x+tile_size]


def generate_pyramid_level(reader, index, dtype, tile_size, level):
    # Convert to float32 because otherwise pyramid_gaussian will convert image to float64
    # which reserves excessive memory
    logger.info("Generate pyramid levels up to %s", level)
    img = read_image_in_chunks(reader, index, dtype)
    original_dtype = img.dtype
    img = skimage.img_as_float32(img, False)

    max_layer = max(max(np.ceil(np.log2(np.array(img.shape) / tile_size))), 0)
    pyramid = skimage.transform.pyramid_gaussian(img, max_layer=max_layer, multichannel=False)
    for layer, layer_img in enumerate(pyramid):
        if layer == level:
            layer_img = skimage.util.dtype.convert(layer_img, original_dtype)
            return layer_img

def set_reader_pyramid_level(level, faas_pyramid=False):
    if not faas_pyramid:
        reader.setResolution(level)
    else:
        reader.setSeries(level)

def get_builtin_pyramid_level_data(index, level, dtype, faas_pyramid=False):
    set_reader_pyramid_level(level, faas_pyramid)
    layer_img = read_image_in_chunks(reader, index, dtype)
    return layer_img

def get_tile(pyramid_img, xi, yi, tile_size):
    return pyramid_img[yi*tile_size:(yi+1)*tile_size, xi*tile_size:(xi+1)*tile_size]


def transform_xml(metadata, id_map):
    ElementTree.register_namespace('', OME_NS)
    xml_root = ElementTree.fromstring(metadata.dumpXML())
    missing = []
    iterfind = lambda m: xml_root.iterfind(f'ome:{m}', {'ome': OME_NS})
    # Replace original image ID attributes with our UUID-based IDs.
    for elt in itertools.chain(iterfind('Image'), iterfind('ImageRef')):
        new_id = id_map.get(elt.attrib['ID'])
        if new_id is not None:
            elt.attrib['ID'] = new_id
        else:
            # Keep a list of elements missing from the map.
            missing.append(elt)
    # Drop the missing elements. This is currently only meant to handle Faas
    # pyramids where we need to drop the subresolution images.
    for elt in missing:
        # We should only ever see Image elements in this list.
        if elt.tag == f'{{{OME_NS}}}Image':
            xml_root.remove(elt)
        else:
            raise ValueError('Unexpected reference to dropped image')
    xml_str = ElementTree.tostring(xml_root, encoding='utf-8')
    return xml_str


def is_faas_pyramid(image_reader):
    """Return True if this is an OME-TIFF containing a 'Faas' pyramid."""
    format_reader = image_reader.getReader()
    format_reader = jnius.cast(IFormatHandler, format_reader)
    if format_reader.getFormat() == 'OME-TIFF':
        # This cast does something to the internals of OMETiffReader that seems
        # to be required to make the getClass call on the next line work. It
        # looks like a bug in jnius.
        ome_tiff_reader = jnius.cast(ImageReader, format_reader).getReader()
        # Peek at the protected 'info' field through reflection.
        field_info = ome_tiff_reader.getClass().getDeclaredField(JString('info'))
        info = field_info.get(ome_tiff_reader)
        tiff_reader = jnius.cast(MinimalTiffReader, info[0][0].reader)
        ifd = tiff_reader.getIFDs().get(0)
        software = ifd.getIFDStringValue(IFD.SOFTWARE)
        if 'Faas' in software:
            return True
    return False

def update_fileset_progress(progress):
    logger.info(f'Progress: {progress}')
    if not debug:
        lmb.invoke(
            FunctionName=set_fileset_complete_arn,
            Payload=str.encode(json.dumps({
                'fileset_uuid': str(fileset_uuid),
                'progress': progress
            }))
        )

class ProgressSubscriber(s3transfer.subscribers.BaseSubscriber):

    def __init__(self, img_id, filename):
        self.subkey = f'{img_id}/{filename}'

    def on_done(self, future, **kwargs):
        print(f'Upload completed: {self.subkey}')

# The wrapper for Field provided by jnius only includes a few methods, and 'get'
# is not one of them. We'll monkey-patch it in here.
jnius.reflect.Field.get = jnius.reflect.JavaMethod(
    '(Ljava/lang/Object;)Ljava/lang/Object;'
)

JString = jnius.autoclass('java.lang.String')
DebugTools = jnius.autoclass('loci.common.DebugTools')
IFormatReader = jnius.autoclass('loci.formats.IFormatReader')
IFormatHandler = jnius.autoclass('loci.formats.IFormatHandler')
MetadataStore = jnius.autoclass('loci.formats.meta.MetadataStore')
ServiceFactory = jnius.autoclass('loci.common.services.ServiceFactory')
OMEXMLService = jnius.autoclass('loci.formats.services.OMEXMLService')
ImageReader = jnius.autoclass('loci.formats.ImageReader')
ChannelSeparator = jnius.autoclass('loci.formats.ChannelSeparator')
ClassList = jnius.autoclass('loci.formats.ClassList')
OMETiffReader = jnius.autoclass('loci.formats.in.OMETiffReader')
MinimalTiffReader = jnius.autoclass('loci.formats.in.MinimalTiffReader')
IFD = jnius.autoclass('loci.formats.tiff.IFD')
OMEXMLMetadataImpl = jnius.autoclass('loci.formats.ome.OMEXMLMetadataImpl')

DebugTools.enableLogging(JString("ERROR"))


import_uuid = pathlib.Path(sys.argv[1])
filename = pathlib.Path(sys.argv[2])
reader_class_name = sys.argv[3]
reader_software = sys.argv[4]
reader_version = sys.argv[5]
bucket = sys.argv[6]
fileset_uuid = pathlib.Path(sys.argv[7])
stack_prefix = os.environ['STACKPREFIX']
stage = os.environ['STAGE']
region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

try:
    debug = os.environ['DEBUG'].upper() == 'TRUE'
except KeyError:
    debug = False

TILE_SIZE = 1024
IMAGE_NAME_LENGTH = 256

s3 = boto3.client('s3')
lmb = boto3.client('lambda')

if not debug:
    ssm = boto3.client('ssm')
    set_fileset_complete_arn = ssm.get_parameter(
        Name=f'/{stack_prefix}/{stage}/api/SetFilesetCompleteLambdaARN'
    )['Parameter']['Value']

file_path = import_uuid.resolve() / filename

single = ClassList(IFormatReader)
for cls in ImageReader.getDefaultReaderClasses().getClasses():
    if cls.getName() == reader_class_name:
        single.addClass(cls)
        break
else:
    raise RuntimeError("unknown reader class: %s" % reader_class_name)

factory = ServiceFactory()
service = jnius.cast(OMEXMLService, factory.getInstance(OMEXMLService))
metadata = service.createOMEXMLMetadata()

reader = ChannelSeparator(ImageReader(single))
reader.setFlattenedResolutions(False)
reader.setMetadataStore(metadata)
# FIXME Workaround for pyjnius #300 via explicit String conversion.
reader.setId(JString(str(file_path)))

# FIXME Handle other data types?
supported_dtypes = {
    'uint8': np.uint8,
    'uint16': np.uint16,
    'uint32': np.uint32
}

# Must cast as implementation class, otherwise pyjnius does not see method getPixelsType 
metadata = jnius.cast(OMEXMLMetadataImpl, metadata)
ome_pixel_type = metadata.getPixelsType(0).value
logger.info("Pixel Type: %s", ome_pixel_type)
try:
    dtype = np.dtype(supported_dtypes[ome_pixel_type])
except KeyError as e:
    msg = f"Pixel type '{ome_pixel_type}' is not supported"
    raise RuntimeError(msg) from None

dtype = dtype.newbyteorder("<" if reader.isLittleEndian() else ">")

# FIXME Consider other file types to support higher-depth pixel formats.
tile_ext = 'tif'
tile_content_type = 'image/zarr'
image_format = 'zarr'
image_compression = 'zstd'
contains_pyramid = reader.getResolutionCount() > 1
series_count = reader.getSeriesCount()
faas_pyramid = False

if is_faas_pyramid(reader):
    # Don't treat subresolutions as separate images in Faas pyramids.
    series_count = 1
    contains_pyramid = True
    faas_pyramid = True

series_digits = len(str(series_count))
logger.info("Image contains pyramid: %s", contains_pyramid)

def mk_name(file_path, n):
    stem = file_path.stem
    if series_count > 1:
        n = str(n).zfill(series_digits)
        suffix = f'[{n}]'
    else:
        suffix = ''
    return stem[:IMAGE_NAME_LENGTH - len(suffix)] + suffix

def count_processed(reader, series_count):
    to_process = 0
    for series in range(series_count):
        reader.setSeries(series)
        width = reader.sizeX
        height = reader.sizeY

        while (width >= TILE_SIZE) and (height >= TILE_SIZE):
            num_tiles = math.ceil(width / TILE_SIZE) * math.ceil(height / TILE_SIZE)
            to_process += reader.sizeC * reader.sizeZ * reader.sizeT * num_tiles
            width = width // 2
            height = height // 2

        num_tiles = math.ceil(width / TILE_SIZE) * math.ceil(height / TILE_SIZE)
        to_process += reader.sizeC * reader.sizeZ * reader.sizeT * num_tiles
    return to_process

def read_image_in_chunks(reader, index, dtype):
    """
    Read image in chunks of rows.
    Large images can have more pixels than Java's array length limit (2^31 − 1)
    Numpy does not have any array limit so it's ok to construct a large numpy array
    and set the values in chunks read from Java.
    """
    JAVA_MAX_ARRAY_SIZE = 2147483639
    num_bytes = reader.sizeY * reader.sizeX * np.dtype(dtype).itemsize
    if num_bytes < JAVA_MAX_ARRAY_SIZE:
        # Image total bytes is under Java's array length limit
        # We can just read the whole image at once
        byte_array = reader.openBytes(index)
        shape = (reader.sizeY, reader.sizeX)
        img = np.frombuffer(byte_array.tostring(), dtype=dtype)
        img = img.reshape(shape)
        return img

    logger.info("Using chunked reading, image bytes: %s", num_bytes)
    CHUNK_WIDTH = reader.sizeX
    CHUNK_HEIGHT = 1000
    img = np.zeros((reader.sizeY, reader.sizeX), dtype)
    chunks_y = math.ceil(reader.sizeY / CHUNK_HEIGHT)
    chunks_x = math.ceil(reader.sizeX / CHUNK_WIDTH)
    for chunk_x in range(chunks_x):
        for chunk_y in range(chunks_y):
            width = CHUNK_WIDTH
            height = CHUNK_HEIGHT
            if chunk_x == chunks_x - 1:
                width = reader.sizeX - (CHUNK_WIDTH * chunk_x)
            if chunk_y == chunks_y - 1:
                height = reader.sizeY - (CHUNK_HEIGHT * chunk_y)

            x = chunk_x * CHUNK_WIDTH
            y = chunk_y * CHUNK_HEIGHT
            logger.debug("Reading bytes from {}:{},{}:{}".format(x, x + width, y, y + height))
            chunk = reader.openBytes(index, x, y, width, height)
            chunk_arr = np.frombuffer(chunk.tostring(), dtype=dtype)
            chunk_arr = chunk_arr.reshape((height, width))
            img[y:y + height, x:x + width] = chunk_arr

    return img


transfer_config = s3transfer.manager.TransferConfig(max_request_queue_size=500, max_submission_queue_size=500)


def handle_tile(c, z, t, level, ty, tx, tile_img, upload_futures):
    global filename, future
    logger.info(f'Tile C={c} level={level} x={tx} y={ty}')
    filename = f'C{c}-T{t}-Z{z}-L{level}-Y{ty}-X{tx}.{tile_ext}'
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', r'.* is a low contrast image', UserWarning,
            '^skimage\.io'
        )
        tifffile.imwrite(buf, tile_img, compress=("ZSTD", 1))
    buf.seek(0)
    tile_key = f'{img_id}/{filename}'
    future = transfer_manager.upload(
        buf, bucket, tile_key, extra_args=upload_args
    )
    upload_futures.append(future)

def get_max_level(width, height, faas_pyramid=False, contains_pyramid=False, tile_size=1024):
    if contains_pyramid:
        return reader.getResolutionCount() if not faas_pyramid else reader.getSeriesCount()
    else:
        levels = 1
        while width > tile_size and height > tile_size:
            width = width // 2
            height = height // 2
            levels += 1
        return levels


with s3transfer.manager.TransferManager(s3, config=transfer_config) as transfer_manager:
    image_id_map = {}
    images = []
    processed = 0
    to_process = count_processed(reader, series_count)
    logger.info("Tiles to process %s", to_process)
    is_rgb = reader.isRGB()
    logger.info("RGB: %s", is_rgb)
    upload_args = dict(ContentType=tile_content_type)

    s3 = s3fs.S3FileSystem(anon=False, client_kwargs=dict(region_name=region))
    compressor = zarr.Blosc(cname='zstd', clevel=3)

    for series in range(series_count):
        reader.setSeries(series)
        rc = range(reader.sizeC)
        rz = range(reader.sizeZ)
        rt = range(reader.sizeT)
        width = reader.sizeX
        height = reader.sizeY
        img_id = str(uuid.uuid4())
        old_xml_id = metadata.getImageID(series)
        image_id_map[old_xml_id] = f'Image:{img_id}'
        logger.info(f'Allocated ID for series {series}: {img_id}')

        s3_store = s3fs.S3Map(root=f"{bucket}/{img_id}", s3=s3, check=False)
        output = zarr.group(store=s3_store, overwrite=True)

        max_level = get_max_level(width, height, faas_pyramid, contains_pyramid, TILE_SIZE)
        logger.info("Max level %s", max_level)

        for level in range(max_level):
            if contains_pyramid:
                set_reader_pyramid_level(level, faas_pyramid)

            arr = output.create(shape=(reader.sizeT, reader.sizeC, reader.sizeZ, reader.sizeY, reader.sizeX),
                                chunks=(1, 1, 1, TILE_SIZE, TILE_SIZE),
                                name=str(level), dtype=dtype, compressor=compressor)
            logger.info("Created zarr array %s", arr.shape)
            for c, z, t in itertools.product(rc, rz, rt):
                start = time.time()
                logger.info("Reading Level=%s T=%s C=%s Z=%s", level, t, c, z)
                index = reader.getIndex(z, c, t)
                if contains_pyramid:
                    level_img = get_builtin_pyramid_level_data(index, level, dtype, faas_pyramid=faas_pyramid)
                else:
                    level_img = generate_pyramid_level(reader, index, dtype, TILE_SIZE, level)

                logger.info("Level shape=%s", level_img.shape)

                tiles_height = math.ceil(level_img.shape[0] / TILE_SIZE)
                tiles_width = math.ceil(level_img.shape[1] / TILE_SIZE)
                for yi in range(tiles_height):
                    for xi in range(tiles_width):
                        tile = get_tile(level_img, xi, yi, TILE_SIZE)
                        logger.info("Tile X=%s Y=%s shape=%s", xi, yi, tile.shape)

                        arr[t, c, z,
                            yi*TILE_SIZE:yi*TILE_SIZE+tile.shape[0],
                            xi*TILE_SIZE:xi*TILE_SIZE+tile.shape[1]] = tile

                        processed += 1

                end = round((time.time() - start) * 1000)
                logger.info(f'Level {level} Channel {c} processed in {end} ms')

                progress = min(math.floor(processed / to_process * 100), 99)
                update_fileset_progress(progress)


        # Add this new image to the list to be attached to this Fileset
        images.append({
            'uuid': img_id,
            'name': mk_name(file_path, series),
            'pyramid_levels': max_level + 1,
            'format': image_format,
            'compression': image_compression,
            'tile_size': TILE_SIZE,
            'rgb': is_rgb,
            'pixel_type': ome_pixel_type
        })

    xml_key = str(fileset_uuid / 'metadata.xml')
    xml_bytes = transform_xml(metadata, image_id_map)
    xml_buf = io.BytesIO(xml_bytes)
    upload_args = dict(ContentType='application/xml')
    future = transfer_manager.upload(
        xml_buf, bucket, xml_key, extra_args=upload_args
    )
    future.result()
    update_fileset_progress(100)

    logger.info('Completing Fileset {} and registering images: {}'.format(
        fileset_uuid,
        ', '.join([image['uuid'] for image in images])
    ))

    if not debug:

        lmb.invoke(
            FunctionName=set_fileset_complete_arn,
            Payload=str.encode(json.dumps({
                'fileset_uuid': str(fileset_uuid),
                'images': images,
                'complete': 'True'
            }))
        )
