# GitHub + VSCode Setup Guide
## For FACEBLUR Project

---

## Step 1 — Create a GitHub Account
If you don't have one:
1. Go to **https://github.com**
2. Click **Sign up**
3. Follow the steps

---

## Step 2 — Create a New Repository

1. On GitHub, click the **+** icon (top right) → **New repository**
2. Fill in:
   - **Repository name:** `faceblur`
   - **Description:** `Automated face censoring application using YOLOv11`
   - **Visibility:** Public or Private (your choice)
   - **DO NOT** check "Add README" — we already have one
3. Click **Create repository**
4. Copy the repository URL shown — looks like:
   ```
   https://github.com/yourusername/faceblur.git
   ```

---

## Step 3 — Install Git on Windows

1. Download from **https://git-scm.com/download/win**
2. Install with default settings
3. Verify installation — open a terminal and run:
   ```bash
   git --version
   ```
   Should print something like `git version 2.44.0`

---

## Step 4 — Install VSCode (if not already)

Download from **https://code.visualstudio.com**

---

## Step 5 — Install GitHub Extension in VSCode

1. Open VSCode
2. Press `Ctrl + Shift + X` to open Extensions
3. Search **GitHub Pull Requests and Issues**
4. Click **Install**

Also install:
- **GitLens** — shows who changed what line and when (very useful)

---

## Step 6 — Sign into GitHub in VSCode

1. Press `Ctrl + Shift + P` → type **Sign in to GitHub**
2. Click **GitHub: Sign In**
3. Browser opens → log into GitHub → authorize VSCode
4. Return to VSCode — you're signed in

---

## Step 7 — Initialize Git in Your Project Folder

Open terminal in VSCode (`Ctrl + ~`) and run:

```bash
# Navigate to your project folder
cd "C:\Users\seoul\Desktop\Joontae\executable\YOLO Setup"

# Initialize git repository
git init

# Set your identity (first time only)
git config --global user.name "Your Name"
git config --global user.email "your@email.com"

# Connect to your GitHub repository
git remote add origin https://github.com/yourusername/faceblur.git
```

---

## Step 8 — First Commit and Push

```bash
# Stage all files
git add .

# Make your first commit
git commit -m "Initial commit - FACEBLUR v1.0"

# Push to GitHub
git push -u origin main
```

Go to your GitHub repo page — all files should now be there.

---

## Step 9 — Daily Workflow in VSCode

VSCode has a built-in Git panel — no need to type commands every time.

### To save changes to GitHub:

1. Make your changes in `face_blur.py` (or any file)
2. Click the **Source Control** icon in the left sidebar (looks like a branch `⑂`)
3. You'll see all changed files listed
4. Hover a file → click **+** to stage it (or click **+** next to "Changes" to stage all)
5. Type a message in the box at the top, e.g. `Fix slider values`
6. Click **✓ Commit**
7. Click **Sync Changes** (or the cloud ↑ icon) to push to GitHub

### Keyboard shortcut:
- `Ctrl + Shift + G` → opens Source Control panel

---

## Step 10 — Create a Release (for distributing the exe)

After building `FACEBLUR_Setup.exe` with `build_installer.bat`:

1. Go to your GitHub repo page
2. Click **Releases** (right sidebar) → **Create a new release**
3. Click **Choose a tag** → type `v1.0` → Create tag
4. Title: `FACEBLUR v1.0`
5. Description: paste your release notes
6. Drag and drop `FACEBLUR_Setup.exe` into the assets section
7. Click **Publish release**

Users can now download directly from:
```
https://github.com/yourusername/faceblur/releases
```

---

## Useful Git Commands (reference)

```bash
# Check status of changed files
git status

# See what changed in a file
git diff face_blur.py

# Pull latest changes from GitHub
git pull

# See commit history
git log --oneline

# Undo changes to a file (before committing)
git checkout face_blur.py

# Create a new branch for a feature
git checkout -b feature/new-feature

# Merge branch back to main
git checkout main
git merge feature/new-feature
```

---

## Recommended Commit Message Format

```
Fix: slider values not updating visually
Add: collapsible sections for parameters
Update: YOLO models to v11 compatible URLs
Remove: inline tips card, replaced with popup
```

Use short, clear messages starting with a verb.

---

*Keep commits small and frequent — one feature or fix per commit.*
