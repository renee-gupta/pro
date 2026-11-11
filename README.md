# Credit Risk & Loan Default Analyser

Multiple logistic regression on LendingClub loan data, with an interactive website.

- `index.html` + `model.js`: the website. The model is trained in your browser (no server needed).
- `analysis.py`: the same pipeline in pandas and scikit-learn (the course version).
- `data/loans.csv`: dataset (loan_status, annual_inc, dti, fico_range_low, loan_amnt).

## Run locally
    python -m http.server 8000     # then open http://localhost:8000
    python analysis.py             # needs: pandas scikit-learn matplotlib

## Publish on GitHub Pages
1. Create a repo and push all files (keep the `data/` folder).
2. Repo Settings > Pages > Source: "Deploy from a branch" > `main` / `(root)` > Save.
3. Your site appears at `https://<username>.github.io/<repo>/`.
