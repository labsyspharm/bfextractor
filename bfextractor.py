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


OME_NS = 'http://www.openmicroscopy.org/Schemas/OME/2016-06'


def tile(img, n):
    for yi, y in enumerate(range(0, img.shape[0], n)):
        for xi, x in enumerate(range(0, img.shape[1], n)):
            yield yi, xi, img[y:y+n, x:x+n]


def build_pyramid(img, n):
    max_layer = max(np.ceil(np.log2(np.array(img.shape) / n)))
    pyramid = skimage.transform.pyramid_gaussian(img, max_layer=max_layer)
    for layer, layer_img in enumerate(pyramid):
        for yi, xi, tile_img in tile(layer_img, n):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', r'Possible precision loss', UserWarning,
                    '^skimage\.util\.dtype'
                )
                tile_img = skimage.util.dtype.convert(tile_img, img.dtype)
            yield layer, yi, xi, tile_img


def transform_xml(metadata, id_map):
    ElementTree.register_namespace('', OME_NS)
    xml_root = ElementTree.fromstring(metadata.dumpXML())
    iterfind = lambda m: xml_root.iterfind(f'ome:{m}', {'ome': OME_NS})
    for elt in itertools.chain(iterfind('Image'), iterfind('ImageRef')):
        elt.attrib['ID'] = image_id_map[elt.attrib['ID']]
    xml_str = ElementTree.tostring(xml_root, encoding='utf-8')
    return xml_str


class ProgressSubscriber(s3transfer.subscribers.BaseSubscriber):

    def __init__(self, img_id, filename):
        self.subkey = f'{img_id}/{filename}'

    def on_done(self, future, **kwargs):
        print(f'Upload completed: {self.subkey}')


JString = jnius.autoclass('java.lang.String')
DebugTools = jnius.autoclass('loci.common.DebugTools')
IFormatReader = jnius.autoclass('loci.formats.IFormatReader')
MetadataStore = jnius.autoclass('loci.formats.meta.MetadataStore')
ServiceFactory = jnius.autoclass('loci.common.services.ServiceFactory')
OMEXMLService = jnius.autoclass('loci.formats.services.OMEXMLService')
ImageReader = jnius.autoclass('loci.formats.ImageReader')
ClassList = jnius.autoclass('loci.formats.ClassList')

DebugTools.enableLogging(JString("ERROR"))


import_uuid = pathlib.Path(sys.argv[1])
filename = pathlib.Path(sys.argv[2])
reader_class_name = sys.argv[3]
bucket = sys.argv[4]
bfu_uuid = pathlib.Path(sys.argv[5])
stack_prefix = os.environ['STACKPREFIX']
stage = os.environ['STAGE']

TILE_SIZE = 1024

s3 = boto3.client('s3')
ssm = boto3.client('ssm')
lmb = boto3.client('lambda')

image_register_arn = ssm.get_parameter(
    Name='/{}/{}/batch/RegisterImageLambdaARN'.format(stack_prefix, stage)
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

reader = ImageReader(single)
reader.setFlattenedResolutions(False)
reader.setMetadataStore(metadata)
# FIXME Workaround for pyjnius #300 via explicit String conversion.
reader.setId(JString(str(file_path)))

# FIXME Implement RGB support.
assert not reader.isRGB(), "RGB images not yet supported"
# FIXME Handle other data types.
assert metadata.getPixelsType(0).value == 'uint16', \
    "Only uint16 images currently supported"
dtype = np.uint16

# FIXME Consider other file types to support higher-depth pixel formats.
tile_ext = 'png'
tile_content_type = 'image/png'


with s3transfer.manager.TransferManager(s3) as transfer_manager:

    upload_futures = []
    image_id_map = {}

    for series in range(reader.getSeriesCount()):

        reader.setSeries(series)
        rc = range(reader.sizeC)
        rz = range(reader.sizeZ)
        rt = range(reader.sizeT)
        img_id = str(uuid.uuid4())
        old_xml_id = metadata.getImageID(series)
        image_id_map[old_xml_id] = f'Image:{img_id}'
        print(f'Allocated ID for series {series}: {img_id}')

        # TODO Better way to get number of levels
        max_level = 0

        for c, z, t in itertools.product(rc, rz, rt):

            index = reader.getIndex(z, c, t)
            byte_array = reader.openBytes(index)
            # FIXME BioFormats seems to return the same size for all series,
            # at least for Metamorph datasets with one different-sized image.
            # Is this a broad BioFormats issue or just that reader?
            shape = (reader.sizeY, reader.sizeX)
            img = np.frombuffer(byte_array.tostring(), dtype=dtype)
            img = img.reshape(shape)

            for level, ty, tx, tile_img in build_pyramid(img, TILE_SIZE):

                if level > max_level:
                    max_level = level

                filename = f'C{c}-T{t}-Z{z}-L{level}-Y{ty}-X{tx}.{tile_ext}'
                buf = io.BytesIO()
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        'ignore', r'.* is a low contrast image', UserWarning,
                        '^skimage\.io'
                    )
                    skimage.io.imsave(buf, tile_img, format_str=tile_ext)
                buf.seek(0)

                tile_key = str(pathlib.Path(img_id) / filename)
                upload_args=dict(ContentType=tile_content_type)

                future = transfer_manager.upload(
                    buf, bucket, tile_key, extra_args=upload_args
                )
                upload_futures.append(future)

        # TODO Do this in another thread and only when it has sucesfully been
        # tiled and uploaded
        print('Registering image {} in BFU {}'.format(img_id, str(bfu_uuid)))
        lmb.invoke(
            FunctionName=image_register_arn,
            Payload=str.encode(json.dumps({
                'bfuUuid': str(bfu_uuid),
                'imageUuid': img_id,
                'pyramidLevels': max_level + 1
            }))
        )

    xml_key = str(bfu_uuid / 'metadata.xml')
    xml_bytes = transform_xml(metadata, image_id_map)
    xml_buf = io.BytesIO(xml_bytes)
    upload_args=dict(ContentType='application/xml')
    future = transfer_manager.upload(
        xml_buf, bucket, xml_key, extra_args=upload_args
    )
    upload_futures.append(future)

    for future in upload_futures:
        future.result()
