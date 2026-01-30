## Resume Evaluator: Supported Upload Formats

The Resume Evaluator supports:
- **PDF** (`.pdf`) — text PDFs
- **Scanned/Image PDFs** (`.pdf`) — via **OCR fallback** (Tesseract)
- **Word DOCX** (`.docx`)
- **Word DOC** (`.doc`) — via **LibreOffice headless conversion** to `.docx`

## Scanned PDFs (OCR)

Scanned/image-only PDFs typically contain **no extractable text**. In that case the app will use OCR.

Requirements:
- Python packages: `pymupdf`, `pytesseract`, `pillow` (in `requirements.txt`)
- System install: **Tesseract OCR**

Windows:
- Install Tesseract OCR and ensure `tesseract.exe` is available.
  - Default path the app can auto-detect: `C:\Program Files\Tesseract-OCR\tesseract.exe`
  - Winget commonly installs per-user at: `%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe`
  - Or add it to PATH
  - Or set `TESSERACT_CMD` to the full path of `tesseract.exe`

Tuning:
- `OCR_MAX_PAGES` (default `8`)
- `OCR_LANG` (default `eng`)
- `ENABLE_PDF_OCR` (default `1`) to enable/disable OCR

## Requirements for `.doc` Support

Legacy `.doc` files require conversion using **LibreOffice**:
- Install **LibreOffice**
- Ensure `soffice.exe` is available (either on PATH or at one of):
  - `C:\Program Files\LibreOffice\program\soffice.exe`
  - `C:\Program Files (x86)\LibreOffice\program\soffice.exe`

If LibreOffice isn’t available, the app will return a clear error asking to install it or convert to `.docx`.

