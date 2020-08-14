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
import s3transfer.manager
import s3transfer.subscribers
import numpy as np
import skimage.io
import skimage.transform
import jnius
import logging, time
import tifffile
import gc

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

OME_NS = 'http://www.openmicroscopy.org/Schemas/OME/2016-06'

def tile(img, n):
    for yi, y in enumerate(range(0, img.shape[0], n)):
        for xi, x in enumerate(range(0, img.shape[1], n)):
            yield yi, xi, img[y:y+n, x:x+n]


def build_pyramid(img, n):
    # Convert to float32 because otherwise pyramid_gaussian will convert image to float64
    # which reserves excessive memory
    original_dtype = img.dtype
    img = skimage.img_as_float32(img, False)

    max_layer = max(max(np.ceil(np.log2(np.array(img.shape) / n))), 0)
    pyramid = skimage.transform.pyramid_gaussian(img, max_layer=max_layer, multichannel=False)
    for layer, layer_img in enumerate(pyramid):
        for yi, xi, tile_img in tile(layer_img, n):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', r'Possible precision loss', UserWarning,
                    '^skimage\.util\.dtype'
                )
                tile_img = skimage.util.dtype.convert(tile_img, original_dtype)
            yield layer, yi, xi, tile_img

def get_pyramid(index, tile_size, dtype, faas_pyramid=False):
    max_layer = reader.getResolutionCount() if not faas_pyramid else reader.getSeriesCount()
    for layer in range(max_layer):
        if not faas_pyramid:
            reader.setResolution(layer)
        else:
            reader.setSeries(layer)

        layer_img = read_image_in_chunks(reader, index, dtype)

        for yi, xi, tile_img in tile(layer_img, tile_size):
            tile_img = skimage.util.dtype.convert(tile_img, dtype)
            yield layer, yi, xi, tile_img


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
try:
    dtype = supported_dtypes[ome_pixel_type]
except KeyError as e:
    msg = f"Pixel type '{ome_pixel_type}' is not supported"
    raise RuntimeError(msg) from None

# FIXME Consider other file types to support higher-depth pixel formats.
tile_ext = 'tif'
tile_content_type = 'image/tiff'
image_format = 'tiff'
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
print("Image contains pyramid: ", contains_pyramid)

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
        to_process += reader.sizeC * reader.sizeZ * reader.sizeT

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

    print("Using chunked reading, image bytes: ", num_bytes)
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
            print("Reading bytes from {}:{},{}:{}".format(x, x + width, y, y + height))
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

def get_max_level(faas_pyramid=False):
    return reader.getResolutionCount() if not faas_pyramid else reader.getSeriesCount()

with s3transfer.manager.TransferManager(s3, config=transfer_config) as transfer_manager:

    image_id_map = {}
    images = []
    processed = 0
    to_process = count_processed(reader, series_count)
    upload_args = dict(ContentType=tile_content_type)

    for series in range(series_count):

        reader.setSeries(series)
        rc = range(reader.sizeC)
        rz = range(reader.sizeZ)
        rt = range(reader.sizeT)
        img_id = str(uuid.uuid4())
        old_xml_id = metadata.getImageID(series)
        image_id_map[old_xml_id] = f'Image:{img_id}'
        print(f'Allocated ID for series {series}: {img_id}')

        # TODO Better way to get number of levels
        max_level = get_max_level(faas_pyramid)

        for c, z, t in itertools.product(rc, rz, rt):
            # FIXME BioFormats seems to return the same size for all series,
            # at least for Metamorph datasets with one different-sized image.
            # Is this a broad BioFormats issue or just that reader?
            upload_futures = []
            start = time.time()
            index = reader.getIndex(z, c, t)

            if not contains_pyramid:
                img = read_image_in_chunks(reader, index, dtype)
                for level, ty, tx, tile_img in build_pyramid(img, TILE_SIZE):
                    handle_tile(c, z, t, level, ty, tx, tile_img, upload_futures)
                    max_level = max(level, max_level)
            else:
                for level, ty, tx, tile_img in get_pyramid(index, TILE_SIZE, dtype, faas_pyramid=faas_pyramid):
                    handle_tile(c, z, t, level, ty, tx, tile_img, upload_futures)

            # Clear all finished uploads, to free up some memory
            finished = [f for f in upload_futures if future.done()]
            for future in finished:
                upload_futures.remove(future)
                del future

            gc.collect()

            end = round((time.time() - start) * 1000)
            logger.info(f'Channel {c} processed in {end} ms')
            processed += 1
            progress = round(processed / to_process * 100)
            update_fileset_progress(progress)


        # Add this new image to the list to be attached to this Fileset
        images.append({
            'uuid': img_id,
            'name': mk_name(file_path, series),
            'pyramid_levels': max_level + 1,
            'format': image_format,
            'compression': image_compression,
            'tile_size': TILE_SIZE
        })

    xml_key = str(fileset_uuid / 'metadata.xml')
    xml_bytes = transform_xml(metadata, image_id_map)
    xml_buf = io.BytesIO(xml_bytes)
    upload_args = dict(ContentType='application/xml')
    future = transfer_manager.upload(
        xml_buf, bucket, xml_key, extra_args=upload_args
    )
    upload_futures.append(future)
    for future in upload_futures:
        future.result()

    print('Completing Fileset {} and registering images: {}'.format(
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
