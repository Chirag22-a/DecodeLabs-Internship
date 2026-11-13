"""Read printed text from an image with OpenCV preprocessing and Tesseract OCR."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = SCRIPT_DIR / "sample_input.png"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"
DEFAULT_MIN_CONFIDENCE = 80.0
WINDOWS_TESSERACT_PATHS = (
    Path.home() / "AppData" / "Local" / "Tesseract-OCR" / "tesseract.exe",
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
)


def configure_tesseract() -> str:
    """Find the Tesseract executable, including its common Windows location."""
    configured_path = os.environ.get("TESSERACT_CMD")
    if configured_path:
        candidate = Path(configured_path.strip('"'))
        if candidate.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            return str(candidate)
        resolved = shutil.which(configured_path)
        if resolved:
            pytesseract.pytesseract.tesseract_cmd = resolved
            return resolved
        raise FileNotFoundError(
            f"TESSERACT_CMD points to a file that does not exist: {configured_path}"
        )

    executable = shutil.which("tesseract")
    if executable:
        pytesseract.pytesseract.tesseract_cmd = executable
        return executable

    for candidate in WINDOWS_TESSERACT_PATHS:
        if candidate.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            return str(candidate)

    raise FileNotFoundError(
        "Tesseract OCR is not installed or not on PATH. Install the Windows "
        "Tesseract engine, then restart PowerShell. See the README for the link."
    )


def read_image(path: Path) -> np.ndarray:
    """Read an image while supporting Windows paths containing Unicode characters."""
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    image_bytes = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read this image format: {path.suffix or '(no extension)'}")
    return image


def deskew(binary_image: np.ndarray) -> np.ndarray:
    """Correct small text rotation angles while leaving blank images untouched."""
    foreground = np.column_stack(np.where(binary_image < 255))
    if len(foreground) < 5:
        return binary_image

    angle = cv2.minAreaRect(foreground.astype(np.float32))[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5 or abs(angle) > 15:
        return binary_image

    height, width = binary_image.shape[:2]
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        binary_image,
        rotation,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale, denoise, adaptively threshold, and deskew."""
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(grayscale, (3, 3), 0)
    thresholded = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return deskew(thresholded)


def recognize_text(
    processed_image: np.ndarray, page_segmentation_mode: int
) -> tuple[str, dict[str, list[str]], list[float]]:
    config = f"--oem 3 --psm {page_segmentation_mode} -c preserve_interword_spaces=1"
    text = pytesseract.image_to_string(processed_image, lang="eng", config=config).strip()
    data = pytesseract.image_to_data(
        processed_image,
        lang="eng",
        config=config,
        output_type=Output.DICT,
    )

    valid_confidences = []
    for word, confidence_text in zip(data["text"], data["conf"]):
        word = word.strip()
        try:
            confidence = float(confidence_text)
        except (TypeError, ValueError):
            continue
        if word and confidence >= 0:
            valid_confidences.append(confidence)
    return text, data, valid_confidences


def draw_recognition_boxes(
    image: np.ndarray,
    data: dict[str, list[str]],
    minimum_confidence: float,
) -> np.ndarray:
    """Draw a confidence-colored rectangle and score around each recognized line."""
    annotated = image.copy()
    lines: dict[tuple[int, int, int], dict[str, object]] = {}
    for index, word in enumerate(data["text"]):
        if not word.strip():
            continue
        try:
            confidence = float(data["conf"][index])
            x = int(data["left"][index])
            y = int(data["top"][index])
            width = int(data["width"][index])
            height = int(data["height"][index])
            line_key = tuple(
                int(data[field][index]) for field in ("block_num", "par_num", "line_num")
            )
        except (TypeError, ValueError):
            continue
        if confidence < 0 or width <= 0 or height <= 0:
            continue

        line = lines.setdefault(
            line_key,
            {"left": x, "top": y, "right": x + width, "bottom": y + height, "scores": []},
        )
        line["left"] = min(int(line["left"]), x)
        line["top"] = min(int(line["top"]), y)
        line["right"] = max(int(line["right"]), x + width)
        line["bottom"] = max(int(line["bottom"]), y + height)
        line["scores"].append(confidence)

    for line in lines.values():
        confidence = sum(line["scores"]) / len(line["scores"])
        image_height, image_width = annotated.shape[:2]
        left = max(1, int(line["left"]) - 8)
        top = max(18, int(line["top"]) - 8)
        right = min(image_width - 2, int(line["right"]) + 8)
        bottom = min(image_height - 2, int(line["bottom"]) + 8)
        color = (42, 155, 95) if confidence >= minimum_confidence else (0, 140, 255)
        cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
        cv2.putText(
            annotated,
            f"{confidence:.0f}%",
            (left, top - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def save_image(path: Path, image: np.ndarray) -> None:
    extension = path.suffix or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise OSError(f"Could not encode output image: {path}")
    encoded.tofile(str(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract printed English text from an image using Tesseract OCR."
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=DEFAULT_IMAGE,
        help="image to read (defaults to the included sample_input.png)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="folder where the preprocessed image, text, and annotated image are saved",
    )
    parser.add_argument(
        "--psm",
        type=int,
        choices=(3, 6, 7, 11),
        default=6,
        help="Tesseract page segmentation mode: 3 automatic, 6 text block, 7 one line, 11 sparse text",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help="minimum average OCR confidence required for a passing result (default: 80)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0 <= args.min_confidence <= 100:
        print("Error: --min-confidence must be between 0 and 100.", file=sys.stderr)
        return 2

    try:
        tesseract_path = configure_tesseract()
        image_path = args.image.expanduser().resolve()
        image = read_image(image_path)
        processed = preprocess_image(image)
        text, data, confidences = recognize_text(processed, args.psm)
        average_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem
        preprocessed_path = output_dir / f"{stem}_preprocessed.png"
        annotated_path = output_dir / f"{stem}_annotated.png"
        text_path = output_dir / f"{stem}_text.txt"

        save_image(preprocessed_path, processed)
        save_image(
            annotated_path,
            draw_recognition_boxes(image, data, args.min_confidence),
        )
        status = "PASS" if confidences and average_confidence >= args.min_confidence else "REVIEW"
        result_text = text if text else "(No text detected.)"
        text_path.write_text(
            f"Recognized text:\n{result_text}\n\n"
            f"Average word confidence: {average_confidence:.1f}%\n"
            f"Required minimum: {args.min_confidence:.1f}%\n"
            f"Validation: {status}\n",
            encoding="utf-8",
        )

        print("IMAGE TEXT RECOGNITION (OCR)")
        print(f"Tesseract: {tesseract_path}")
        print(f"Input image: {image_path}")
        print("\nRecognized text:")
        print(result_text)
        print(f"\nAverage word confidence: {average_confidence:.1f}%")
        print(f"Confidence threshold: {args.min_confidence:.1f}%")
        print(f"Validation: {status}")
        print(f"Preprocessed image: {preprocessed_path}")
        print(f"Annotated image: {annotated_path}")
        print(f"Text result: {text_path}")
        return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
