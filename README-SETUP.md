# GitHub Pages fix — Grammar Scoring Project

## Why the old URL showed 404
GitHub Pages needs an `index.html` in the folder configured as the publishing source. The repository root shown in the supplied listing did not contain one.

## Files in this package
- `index.html` — project landing page
- `style.css` — responsive styling
- `script.js` — small client-side enhancement

## Publish these files
1. Download and extract this ZIP.
2. Open your GitHub repository: https://github.com/saiyogesh-1244/grammar_scoring_project
3. Choose **Add file → Upload files**.
4. Upload `index.html`, `style.css`, and `script.js` to the repository's **main/root folder** (not inside `src`).
5. Commit the changes to the `main` branch.
6. Open **Settings → Pages**.
7. Under **Build and deployment**, choose **Deploy from a branch**.
8. Select branch **main** and folder **/(root)**, then click **Save**.
9. Wait a few minutes and open: https://saiyogesh-1244.github.io/grammar_scoring_project/

## Important: Python backend
GitHub Pages serves static websites only. This landing page does not execute your Python grammar-scoring model. To make actual grammar scoring work online, the Python application must be deployed separately (for example, as a Python API), then this page can call that API. The exact integration depends on the contents of `src` and the model files.
