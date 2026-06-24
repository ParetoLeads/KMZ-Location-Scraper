# GitHub Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a public GitHub repo for `paretoleads` and push the existing local codebase to it.

**Architecture:** Simple git remote setup — add `.gitignore`, create the repo via `gh` CLI, push main branch.

**Tech Stack:** Git, GitHub CLI (`gh`)

## Global Constraints

- Repo name: `kmz-location-scraper`
- GitHub account: `ParetoLeads` (confirmed via `gh auth status`)
- Visibility: public
- Default branch: `main` (already exists)

---

### Task 1: Create .gitignore and push to GitHub

**Files:**
- Create: `.gitignore`

**Interfaces:**
- Produces: GitHub repo at `https://github.com/ParetoLeads/kmz-location-scraper`

- [ ] **Step 1: Create .gitignore**

```text
# Python
__pycache__/
*.py[cod]
*.pyo
.venv/
venv/
env/
*.egg-info/
dist/
build/

# Streamlit secrets (never commit)
.streamlit/secrets.toml

# Environment files
.env
.env.local

# macOS
.DS_Store

# IDE
.idea/
.vscode/

# Test outputs
*.xlsx
*.kmz
!Boston\ Area.kmz
```

Write this to `.gitignore` in the project root.

- [ ] **Step 2: Stage and verify nothing sensitive is tracked**

Run: `git status`

Expected: `.gitignore` shown as new file. Verify no `.env` or `secrets.toml` files appear.

- [ ] **Step 3: Create the GitHub repo**

Run:
```bash
gh repo create ParetoLeads/kmz-location-scraper --public --description "KMZ boundary to OSM locations with AI population estimates"
```

Expected output: `✓ Created repository ParetoLeads/kmz-location-scraper on GitHub`

- [ ] **Step 4: Add remote origin**

Run:
```bash
git remote add origin https://github.com/ParetoLeads/kmz-location-scraper.git
```

If remote already exists: `git remote set-url origin https://github.com/ParetoLeads/kmz-location-scraper.git`

- [ ] **Step 5: Commit .gitignore and push**

Run:
```bash
git add .gitignore
git commit -m "chore: add .gitignore"
git push -u origin main
```

Expected: Branch `main` pushed. Repo visible at `https://github.com/ParetoLeads/kmz-location-scraper`.
