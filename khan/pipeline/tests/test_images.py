import pytest
from astropy.io import fits
import numpy as np
from numpy.testing import assert_array_equal

from khan.pipeline.images import parse_mosaic_detector_slice, \
    determine_detector_layout, get_mosaic_detector_corner_coordinates, \
    reformat_observers, CCDImage

np.random.seed(1701)


class TestParseMosaicDetectorSlice:

    @pytest.fixture
    def good_slice_string(self):
        yield '[5:683,1:4096]'

    def test_correct_slice_returns_correct_slice(self, good_slice_string):
        assert parse_mosaic_detector_slice(good_slice_string) == \
               (slice(4, 683, 1), slice(0, 4096, 1))


class TestDetermineDetectorLayout:

    @pytest.fixture
    def legacy_hdul(self):
        yield fits.HDUList([fits.PrimaryHDU()])

    @pytest.fixture
    def mosaic_hdul(self):
        yield fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(),
                            fits.ImageHDU(), fits.ImageHDU()])

    @pytest.fixture
    def one_detector_mosaic_hdul(self):
        yield fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU()])

    @pytest.fixture
    def two_detector_mosaic_hdul(self):
        yield fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(),
                            fits.ImageHDU()])

    @pytest.fixture
    def four_detector_mosaic_hdul(self):
        yield fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(),
                            fits.ImageHDU(), fits.ImageHDU(), fits.ImageHDU()])

    def test_if_legacy_identified(self, legacy_hdul):
        assert determine_detector_layout(legacy_hdul) == 'legacy'

    def test_if_mosaic_identified(self, mosaic_hdul):
        assert determine_detector_layout(mosaic_hdul) == 'mosaic'

    def test_if_one_detector_mosaic_raises_exception(self,
                                                     one_detector_mosaic_hdul):
        with pytest.raises(Exception):
            determine_detector_layout(one_detector_mosaic_hdul)

    def test_if_two_detector_mosaic_raises_exception(self,
                                                     two_detector_mosaic_hdul):
        with pytest.raises(Exception):
            determine_detector_layout(two_detector_mosaic_hdul)

    def test_if_four_detector_mosaic_raises_exception(
            self, four_detector_mosaic_hdul):
        with pytest.raises(Exception):
            determine_detector_layout(four_detector_mosaic_hdul)


class TestGetMosaicDetectorCornerCoordinates:

    @pytest.fixture
    def image_header_whole_pixel(self):
        yield fits.Header({'CRVAL1G': 4141.0, 'CRVAL2G': 1.0})

    @pytest.fixture
    def image_header_half_pixel(self):
        yield fits.Header({'CRVAL1G': 4141.5, 'CRVAL2G': 0.5})

    def test_correct_position_with_2x_spatial_binning(
            self, image_header_whole_pixel):
        assert assert_array_equal(
            get_mosaic_detector_corner_coordinates(
                image_header_whole_pixel, binning=np.array([2, 1])),
            np.array([1047, 0])) is None

    def test_correct_position_with_2x_spatial_binning_and_decimal_coordinate(
            self, image_header_half_pixel):
        assert assert_array_equal(
            get_mosaic_detector_corner_coordinates(
                image_header_half_pixel, binning=np.array([2, 1])),
            np.array([1047, 0])) is None

    def test_correct_position_with_3x_spatial_binning(
            self, image_header_whole_pixel):
        assert assert_array_equal(
            get_mosaic_detector_corner_coordinates(
                image_header_whole_pixel, binning=np.array([3, 1])),
            np.array([698, 0])) is None

    def test_correct_position_with_3x_spatial_binning_and_decimal_coordinate(
            self, image_header_half_pixel):
        assert assert_array_equal(
            get_mosaic_detector_corner_coordinates(
                image_header_half_pixel, binning=np.array([3, 1])),
            np.array([698, 0])) is None


class TestReformatObservers:

    @pytest.fixture
    def observers_with_commas_no_spaces(self):
        yield 'de Kleer,Milby,Camarca,Schmidt,Brown'

    @pytest.fixture
    def observers_with_commas_and_spaces(self):
        yield 'de Kleer, Milby, Camarca, Schmidt, Brown'

    def test_observers_with_commas_no_spaces_returns_with_spaces(
            self, observers_with_commas_no_spaces):
        assert reformat_observers(observers_with_commas_no_spaces) == \
               'de Kleer, Milby, Camarca, Schmidt, Brown'

    def test_observers_with_commas_and_spaces_returns_unchanged(
            self, observers_with_commas_and_spaces):
        assert reformat_observers(observers_with_commas_and_spaces) == \
               observers_with_commas_and_spaces


class TestCCDImage:

    @pytest.fixture
    def sample_data(self):
        yield np.random.rand(2, 3)

    @pytest.fixture
    def sample_anc(self):
        yield {'test': 0.0}

    def test_if_data_type_is_array(self, sample_data, sample_anc):
        assert type(CCDImage(sample_data, sample_anc).data) is np.ndarray

    def test_if_data_matches_expected_array(self, sample_data, sample_anc):
        assert assert_array_equal(CCDImage(sample_data, sample_anc).data,
                                  sample_data) is None

    def test_if_ancillary_type_is_dict(self, sample_data, sample_anc):
        assert type(CCDImage(sample_data, sample_anc).anc) is dict

    def test_if_ancillary_information_accessible(
            self, sample_data, sample_anc):
        assert CCDImage(sample_data, sample_anc).anc['test'] == 0.0
