
"""
qr_reader.py

Small command-line app that detects and decodes QR codes in images using OpenCV.

Usage examples:

    # Basic: read and print QR content
    python3 qr_reader.py qr.png

    # Save an annotated copy of the image
    python3 qr_reader.py qr.png --save-annotated annotated_qr.png

Requirements:
    - Python 3.x
    - OpenCV for Python:  pip install opencv-python
"""

import argparse
import os
import sys
from typing import Optional, Tuple, List

import cv2  # OpenCV library


def read_qr_from_image(
    image_path: str,
    show: bool = False,
    save_annotated: bool = False,
    annotated_output_path: Optional[str] = None,
) -> Tuple[List[str], "cv2.Mat"]:
    """
    Detect and decode QR code(s) from an image using OpenCV.

    Parameters
    ----------
    image_path : str
        Path to the input image file (PNG, JPG, etc.)
    save_annotated : bool, optional
        If True, save an annotated copy of the image (with QR bounding box)
        to 'annotated_output_path' (or a default name if not provided).
    annotated_output_path : str or None, optional
        Path where the annotated image should be saved. If None and
        save_annotated is True, a default name is generated based on
        the input filename.

    Returns
    -------
    decoded_texts : list of str
        A list of decoded QR contents. For simple images with a single
        QR code, this list will usually have length 1.
    annotated_img : cv2.Mat
        The image with bounding boxes drawn around any detected QR codes.
        This is the same image that will be shown/saved if requested.

    Raises
    ------
    FileNotFoundError
        If the input image does not exist or cannot be loaded.
    """

    # --- Step 1: Check that the image exists on disk ------------------------
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # --- Step 2: Load the image with OpenCV ---------------------------------
    # cv2.imread returns a NumPy array (the image) or None if it fails.
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to load image (unsupported format?): {image_path}")

    # We'll work on a copy for drawing, so the original stays unchanged.
    annotated_img = img.copy()

    # --- Step 3: Create the QRCodeDetector object ---------------------------
    # This is OpenCV's built-in QR detector/decoder.
    detector = cv2.QRCodeDetector()

    # --- Step 4: Try to detect and decode multiple QR codes first (if supported) --
    decoded_texts: List[str] = []

    # Not all OpenCV builds have detectAndDecodeMulti, so we guard with hasattr.
    if hasattr(detector, "detectAndDecodeMulti"):
        # detectAndDecodeMulti returns:
        #   retval (bool), decoded_info (list of str),
        #   points (list of corner points), straight_qrcode (ignored here)
        retval, decoded_info, points, _ = detector.detectAndDecodeMulti(img)

        if retval and decoded_info is not None:
            # At least one QR code was detected and decoded.
            for text, pts in zip(decoded_info, points):
                if text:  # text may be empty if detection failed
                    decoded_texts.append(text)

                    # pts is a 4x2 array of corner points (float32).
                    # We convert to int for drawing rectangles.
                    pts = pts.astype(int)

                    # Draw lines between the 4 points to show the QR boundary.
                    for i in range(4):
                        pt1 = tuple(pts[i])
                        pt2 = tuple(pts[(i + 1) % 4])
                        cv2.line(annotated_img, pt1, pt2, (0, 255, 0), 2)

            # If we found at least one decoded text, we can skip the single-code path.
            if decoded_texts:
                # Optionally show/save results.
                if show:
                    _show_image(annotated_img, window_title="QR Result (Multi)")

                if save_annotated:
                    _save_annotated_image(
                        annotated_img,
                        image_path,
                        annotated_output_path,
                        suffix="_annotated_multi",
                    )

                return decoded_texts, annotated_img

    # --- Step 5: Fallback: try single QR detection/decoding -----------------
    # detectAndDecode (singular) returns:
    #   data (str), points (corner points), straight_qrcode (ignored)
    data, points, _ = detector.detectAndDecode(img)

    if data:
        # We got a decoded string: that means a QR code was found.
        decoded_texts.append(data)

        if points is not None:
            pts = points[0].astype(int)  # shape is (1, 4, 2) -> (4, 2)
            for i in range(4):
                pt1 = tuple(pts[i])
                pt2 = tuple(pts[(i + 1) % 4])
                cv2.line(annotated_img, pt1, pt2, (0, 0, 255), 2)  # red box

    # --- Step 6: Optionally show and/or save the annotated image ------------
    if show:
        _show_image(annotated_img, window_title="QR Result")

    if save_annotated:
        _save_annotated_image(
            annotated_img,
            image_path,
            annotated_output_path,
            suffix="_annotated",
        )

    # Return the list (possibly empty) of decoded QR contents.
    return decoded_texts, annotated_img


def _show_image(img: "cv2.Mat", window_title: str = "Image") -> None:
    """
    Display an image in a resizable OpenCV window.

    Parameters
    ----------
    img : cv2.Mat
        The image to show.
    window_title : str
        Title of the window.
    """
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.imshow(window_title, img)
    # Wait indefinitely until any key is pressed, then close the window.
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _save_annotated_image(
    img: "cv2.Mat",
    original_path: str,
    explicit_output_path: Optional[str],
    suffix: str = "_annotated",
) -> str:
    """
    Helper function to save an annotated image with a bounding box.

    Parameters
    ----------
    img : cv2.Mat
        The image to save.
    original_path : str
        Path of the original input image file. Used to create a default
        output filename if 'explicit_output_path' is None.
    explicit_output_path : str or None
        If supplied, this exact path will be used to save the image.
    suffix : str
        Suffix to append to the base filename, before the extension.

    Returns
    -------
    output_path : str
        Full path of the written file.
    """
    if explicit_output_path is not None:
        output_path = explicit_output_path
    else:
        # Generate a default filename based on original image name.
        base, ext = os.path.splitext(original_path)
        if not ext:
            ext = ".png"
        output_path = f"{base}{suffix}{ext}"

    # Actually write the image to disk.
    success = cv2.imwrite(output_path, img)
    if not success:
        print(f"[!] Failed to save annotated image to: {output_path}", file=sys.stderr)
    else:
        print(f"[+] Annotated image saved to: {output_path}")

    return output_path


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the QR reader app.

    Returns
    -------
    argparse.Namespace
        Object with attributes corresponding to the CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Detect and decode QR code(s) from an image using OpenCV."
    )

    parser.add_argument(
        "image",
        help="Path to the input image file containing a QR code.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the image with detected QR code boundaries in a window.",
    )

    parser.add_argument(
        "--save-annotated",
        metavar="OUTPUT",
        nargs="?",
        const="",  # If flag is given without a value, we'll auto-generate a name.
        help=(
            "Save an annotated copy of the image (with QR bounding box). "
            "If OUTPUT is omitted, a default name is generated."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point: parse CLI arguments, run the QR detection,
    and print results to the terminal.
    """
    args = parse_args()

    image_path = args.image

    # Determine if we should save an annotated version, and which filename to use.
    save_annotated = args.save_annotated is not None
    annotated_output_path = None
    if save_annotated:
        # If user supplied a path, use it. If they just used --save-annotated
        # with no filename, we leave None and let _save_annotated_image choose.
        if args.save_annotated.strip():
            annotated_output_path = args.save_annotated

    try:
        decoded_texts, _ = read_qr_from_image(
            image_path=image_path,
            show=args.show,
            save_annotated=save_annotated,
            annotated_output_path=annotated_output_path,
        )
    except FileNotFoundError as e:
        print(f"[!] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[!] Unexpected error: {e}")
        sys.exit(1)

    # Print results
    if not decoded_texts:
        print("[*] No QR code detected.")
    else:
        print("[+] Detected QR code(s):")
        for i, text in enumerate(decoded_texts, start=1):
            print(f"    #{i}: {text}")


if __name__ == "__main__":
    main()
