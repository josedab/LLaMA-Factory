# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile

import pytest

from llamafactory.data import Role
from llamafactory.data.converter import get_dataset_converter
from llamafactory.data.parser import DatasetAttr
from llamafactory.hparams import DataArguments


def test_alpaca_converter():
    dataset_attr = DatasetAttr("hf_hub", "llamafactory/tiny-supervised-dataset")
    data_args = DataArguments()
    example = {
        "instruction": "Solve the math problem.",
        "input": "3 + 4",
        "output": "The answer is 7.",
    }
    dataset_converter = get_dataset_converter("alpaca", dataset_attr, data_args)
    assert dataset_converter(example) == {
        "_prompt": [{"role": Role.USER.value, "content": "Solve the math problem.\n3 + 4"}],
        "_response": [{"role": Role.ASSISTANT.value, "content": "The answer is 7."}],
        "_system": "",
        "_tools": "",
        "_images": None,
        "_videos": None,
        "_audios": None,
    }


def test_sharegpt_converter():
    dataset_attr = DatasetAttr("hf_hub", "llamafactory/tiny-supervised-dataset")
    data_args = DataArguments()
    example = {
        "conversations": [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": "Solve the math problem.\n3 + 4"},
            {"from": "gpt", "value": "The answer is 7."},
        ]
    }
    dataset_converter = get_dataset_converter("sharegpt", dataset_attr, data_args)
    assert dataset_converter(example) == {
        "_prompt": [{"role": Role.USER.value, "content": "Solve the math problem.\n3 + 4"}],
        "_response": [{"role": Role.ASSISTANT.value, "content": "The answer is 7."}],
        "_system": "You are a helpful assistant.",
        "_tools": "",
        "_images": None,
        "_videos": None,
        "_audios": None,
    }


# Tests for refactored media helper functions (RFC-0005)


class TestNormalizeMediaInput:
    """Tests for _normalize_media_input helper function."""

    def test_normalize_none_input(self):
        """Test that None input returns None."""
        dataset_attr = DatasetAttr("hf_hub", "test-dataset")
        data_args = DataArguments()
        converter = get_dataset_converter("alpaca", dataset_attr, data_args)
        assert converter._normalize_media_input(None) is None

    def test_normalize_single_item(self):
        """Test that a single item is wrapped in a list."""
        dataset_attr = DatasetAttr("hf_hub", "test-dataset")
        data_args = DataArguments()
        converter = get_dataset_converter("alpaca", dataset_attr, data_args)
        assert converter._normalize_media_input("/path/to/image.jpg") == ["/path/to/image.jpg"]

    def test_normalize_empty_list(self):
        """Test that an empty list returns None."""
        dataset_attr = DatasetAttr("hf_hub", "test-dataset")
        data_args = DataArguments()
        converter = get_dataset_converter("alpaca", dataset_attr, data_args)
        assert converter._normalize_media_input([]) is None

    def test_normalize_list_returns_copy(self):
        """Test that a list input returns a copy."""
        dataset_attr = DatasetAttr("hf_hub", "test-dataset")
        data_args = DataArguments()
        converter = get_dataset_converter("alpaca", dataset_attr, data_args)
        original = ["/path/1.jpg", "/path/2.jpg"]
        result = converter._normalize_media_input(original)
        assert result == original
        assert result is not original  # Should be a copy

    def test_normalize_bytes_input(self):
        """Test that bytes input is wrapped in a list."""
        dataset_attr = DatasetAttr("hf_hub", "test-dataset")
        data_args = DataArguments()
        converter = get_dataset_converter("alpaca", dataset_attr, data_args)
        bytes_data = b"image_data"
        assert converter._normalize_media_input(bytes_data) == [bytes_data]


class TestShouldResolvePaths:
    """Tests for _should_resolve_paths helper function."""

    def test_should_resolve_for_file_load(self):
        """Test that paths should be resolved when loading from file."""
        dataset_attr = DatasetAttr("file", "test-dataset")
        data_args = DataArguments()
        converter = get_dataset_converter("alpaca", dataset_attr, data_args)
        assert converter._should_resolve_paths() is True

    def test_should_resolve_for_script_load(self):
        """Test that paths should be resolved when loading from script."""
        dataset_attr = DatasetAttr("script", "test-dataset")
        data_args = DataArguments()
        converter = get_dataset_converter("alpaca", dataset_attr, data_args)
        assert converter._should_resolve_paths() is True

    def test_should_not_resolve_for_hf_hub(self):
        """Test that paths should not be resolved when loading from HuggingFace Hub."""
        dataset_attr = DatasetAttr("hf_hub", "test-dataset")
        data_args = DataArguments()
        converter = get_dataset_converter("alpaca", dataset_attr, data_args)
        assert converter._should_resolve_paths() is False


class TestResolveMediaPaths:
    """Tests for _resolve_media_paths and related helper functions."""

    def test_resolve_empty_list(self):
        """Test that an empty list returns as-is."""
        dataset_attr = DatasetAttr("file", "test-dataset")
        data_args = DataArguments()
        converter = get_dataset_converter("alpaca", dataset_attr, data_args)
        assert converter._resolve_media_paths([]) == []

    def test_resolve_bytes_returns_as_is(self):
        """Test that bytes items are returned as-is."""
        dataset_attr = DatasetAttr("file", "test-dataset")
        data_args = DataArguments()
        converter = get_dataset_converter("alpaca", dataset_attr, data_args)
        bytes_data = [b"image1", b"image2"]
        assert converter._resolve_media_paths(bytes_data) == bytes_data


class TestResolveSinglePath:
    """Tests for _resolve_single_path helper function."""

    def test_resolve_existing_file(self):
        """Test that existing file paths are resolved to full path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = os.path.join(tmpdir, "test_image.jpg")
            with open(test_file, "w") as f:
                f.write("test")

            dataset_attr = DatasetAttr("file", "test-dataset")
            data_args = DataArguments(media_dir=tmpdir)
            converter = get_dataset_converter("alpaca", dataset_attr, data_args)

            result = converter._resolve_single_path("test_image.jpg")
            assert result == test_file

    def test_resolve_nonexistent_file(self):
        """Test that nonexistent file paths return original path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_attr = DatasetAttr("file", "test-dataset")
            data_args = DataArguments(media_dir=tmpdir)
            converter = get_dataset_converter("alpaca", dataset_attr, data_args)

            result = converter._resolve_single_path("nonexistent.jpg")
            assert result == "nonexistent.jpg"


class TestResolveFlatMediaPaths:
    """Tests for _resolve_flat_media_paths helper function."""

    def test_resolve_multiple_paths(self):
        """Test resolving multiple media paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            files = ["image1.jpg", "image2.jpg"]
            for fname in files:
                with open(os.path.join(tmpdir, fname), "w") as f:
                    f.write("test")

            dataset_attr = DatasetAttr("file", "test-dataset")
            data_args = DataArguments(media_dir=tmpdir)
            converter = get_dataset_converter("alpaca", dataset_attr, data_args)

            result = converter._resolve_flat_media_paths(files)
            expected = [os.path.join(tmpdir, f) for f in files]
            assert result == expected


class TestResolveNestedMediaPaths:
    """Tests for _resolve_nested_media_paths helper function (video frames)."""

    def test_resolve_nested_paths(self):
        """Test resolving nested lists of media paths for video frames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files for two videos with multiple frames each
            frames_video1 = ["v1_frame1.jpg", "v1_frame2.jpg"]
            frames_video2 = ["v2_frame1.jpg", "v2_frame2.jpg"]
            all_frames = frames_video1 + frames_video2

            for fname in all_frames:
                with open(os.path.join(tmpdir, fname), "w") as f:
                    f.write("test")

            dataset_attr = DatasetAttr("file", "test-dataset")
            data_args = DataArguments(media_dir=tmpdir)
            converter = get_dataset_converter("alpaca", dataset_attr, data_args)

            result = converter._resolve_nested_media_paths([frames_video1, frames_video2])
            expected = [
                [os.path.join(tmpdir, f) for f in frames_video1],
                [os.path.join(tmpdir, f) for f in frames_video2],
            ]
            assert result == expected


class TestFindMediasIntegration:
    """Integration tests for the refactored _find_medias function."""

    def test_find_medias_none(self):
        """Test that None input returns None."""
        dataset_attr = DatasetAttr("hf_hub", "test-dataset")
        data_args = DataArguments()
        converter = get_dataset_converter("alpaca", dataset_attr, data_args)
        assert converter._find_medias(None) is None

    def test_find_medias_empty_list(self):
        """Test that empty list returns None."""
        dataset_attr = DatasetAttr("hf_hub", "test-dataset")
        data_args = DataArguments()
        converter = get_dataset_converter("alpaca", dataset_attr, data_args)
        assert converter._find_medias([]) is None

    def test_find_medias_single_item_hf_hub(self):
        """Test single item from HuggingFace Hub (no path resolution)."""
        dataset_attr = DatasetAttr("hf_hub", "test-dataset")
        data_args = DataArguments()
        converter = get_dataset_converter("alpaca", dataset_attr, data_args)
        result = converter._find_medias("/absolute/path/image.jpg")
        assert result == ["/absolute/path/image.jpg"]

    def test_find_medias_list_hf_hub(self):
        """Test list of items from HuggingFace Hub (no path resolution)."""
        dataset_attr = DatasetAttr("hf_hub", "test-dataset")
        data_args = DataArguments()
        converter = get_dataset_converter("alpaca", dataset_attr, data_args)
        paths = ["/path/1.jpg", "/path/2.jpg"]
        result = converter._find_medias(paths)
        assert result == paths

    def test_find_medias_with_path_resolution(self):
        """Test that paths are resolved when loading from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            test_file = "test.jpg"
            with open(os.path.join(tmpdir, test_file), "w") as f:
                f.write("test")

            dataset_attr = DatasetAttr("file", "test-dataset")
            data_args = DataArguments(media_dir=tmpdir)
            converter = get_dataset_converter("alpaca", dataset_attr, data_args)

            result = converter._find_medias(test_file)
            assert result == [os.path.join(tmpdir, test_file)]

    def test_find_medias_nested_video_frames(self):
        """Test processing nested lists for video frames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            frames = [["v1_f1.jpg", "v1_f2.jpg"], ["v2_f1.jpg"]]
            for frame_list in frames:
                for fname in frame_list:
                    with open(os.path.join(tmpdir, fname), "w") as f:
                        f.write("test")

            dataset_attr = DatasetAttr("file", "test-dataset")
            data_args = DataArguments(media_dir=tmpdir)
            converter = get_dataset_converter("alpaca", dataset_attr, data_args)

            result = converter._find_medias(frames)
            expected = [
                [os.path.join(tmpdir, f) for f in frames[0]],
                [os.path.join(tmpdir, f) for f in frames[1]],
            ]
            assert result == expected

    def test_find_medias_bytes_passthrough(self):
        """Test that bytes data passes through without modification."""
        dataset_attr = DatasetAttr("file", "test-dataset")
        data_args = DataArguments()
        converter = get_dataset_converter("alpaca", dataset_attr, data_args)
        bytes_data = b"image_bytes"
        result = converter._find_medias(bytes_data)
        assert result == [bytes_data]
