#!/env/python
import subprocess
import sys
import zarr
import os
from xml.etree import ElementTree

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def assert_shape(arr, shape):
    assert arr.shape == shape, f"Expected shape {shape} but was {arr.shape}"

def run_test(compose_file):
    docker_args = ['docker-compose', '-f', compose_file, 'up']
    completed = subprocess.run(docker_args)
    if completed.returncode != 0:
        raise ValueError("ERROR - bfextractor returned " + completed.returncode)

def validate_metadata(metadata_path):
    assert os.path.exists(metadata_path), "metadata.xml does not exist"
    with open(metadata_path, "r") as metadata_file:
        xml = metadata_file.read()
        root = ElementTree.fromstring(xml)

# BF6 pyramid built-in
def test_bf6_pyramid():
    run_test('docker-compose-test-bf6-pyramid.yml')
    validate_metadata("./test_output/bf6_pyramid/metadata.xml")

    zarr_store = zarr.DirectoryStore('test_output/bf6_pyramid')
    group = zarr.open(zarr_store)
    assert len(group) == 2, "Zarr group should have 2 arrays"
    level0 = group["0"]
    level1 = group["1"]
    assert level0.shape == (1, 3, 1, 2048, 2048)
    assert level0.chunks == (1, 1, 1, 1024, 1024)
    assert level1.shape == (1, 3, 1, 1024, 1024)
    assert level1.chunks == (1, 1, 1, 1024, 1024)


# BF6 no pyramid
def test_bf6_non_pyramid():
    run_test('docker-compose-test-bf6-non-pyramid.yml')
    validate_metadata("./test_output/bf6_non_pyramid/metadata.xml")

    zarr_store = zarr.DirectoryStore('test_output/bf6_non_pyramid')
    group = zarr.open(zarr_store)
    assert len(group) == 4, "Zarr group should have 4 arrays"
    level0 = group["0"]
    level1 = group["1"]
    level2 = group["2"]
    level3 = group["3"]
    assert_shape(level0, (1, 4, 1, 1256, 5000))
    assert level0.chunks == (1, 1, 1, 1024, 1024)
    assert_shape(level1, (1, 4, 1, 628, 2500))
    assert level1.chunks == (1, 1, 1, 1024, 1024)
    assert_shape(level2, (1, 4, 1, 314, 1250))
    assert level1.chunks == (1, 1, 1, 1024, 1024)
    assert_shape(level3, (1, 4, 1, 157, 625))
    assert level1.chunks == (1, 1, 1, 1024, 1024)

# BF5 no pyramid
def test_bf5_faas_pyramid():
    run_test('docker-compose-test-bf5-pyramid.yml')
    validate_metadata("./test_output/bf5_pyramid/metadata.xml")

    zarr_store = zarr.DirectoryStore('test_output/bf5_pyramid')
    group = zarr.open(zarr_store)
    assert len(group) == 2, "Zarr group should have 2 arrays"
    level0 = group["0"]
    level1 = group["1"]
    assert_shape(level0, (1, 3, 1, 2048, 2048))
    assert level0.chunks == (1, 1, 1, 1024, 1024)
    assert_shape(level1, (1, 3, 1, 1024, 1024))
    assert level1.chunks == (1, 1, 1, 1024, 1024)


if __name__ == "__main__":
    test_bf5_faas_pyramid()
    test_bf6_non_pyramid()
    test_bf6_pyramid()
