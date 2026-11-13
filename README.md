# DD2360-assignment-2

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tattack99/DD2360-assignment-2/blob/main/colab.ipynb)

Everything runs in Google Colab, so you don't need to install anything locally.

## One-time setup

### 1. Accept the repo invite
Accept the GitHub invitation to `tattack99/DD2360-assignment-2`, which arrives by email or under your GitHub notifications. The repo is private, so nothing below works until you've accepted.

### 2. Get a GitHub token
The notebook clones the repo with your token. Use a **classic** token with the `repo` scope:

- **Already have a classic token with `repo`?** You can reuse it.
- **Otherwise**, go to GitHub → Settings → Developer settings → Personal access tokens → **Tokens (classic)** → *Generate new token (classic)*, tick **`repo`**, and copy the token (it starts with `ghp_`).

> Fine-grained tokens won't work here. They can't access a repo owned by another user that you're only a collaborator on.

### 3. Open the notebook in Colab
**Quickest way:** [open `colab.ipynb` in Colab](https://colab.research.google.com/github/tattack99/DD2360-assignment-2/blob/main/colab.ipynb). Authorize GitHub if Colab asks; it needs that to read the private repo.

If the link doesn't work, open the notebook manually:
1. Go to [colab.research.google.com](https://colab.research.google.com) → **File → Open notebook → GitHub** tab.
2. Tick **Include private repos** and authorize GitHub if asked.
3. Pick `tattack99/DD2360-assignment-2`, branch `main`, then **`colab.ipynb`**.

### 4. Add your token as a Colab secret
1. Click the **🔑 key icon** (Secrets) in the left sidebar.
2. **Add new secret** with:
   - **Name:** `GH_TOKEN_DD2360_ASSIGNMENT_2` (exactly this)
   - **Value:** your token
3. Turn on **Notebook access** for it.

The secret is tied to your Google account, so you only add it once.

## Every session

1. **Runtime → Change runtime type → T4 GPU** (only needed for GPU/CUDA work).
2. **Runtime → Run all.**

The first cell clones the repo, pulls the latest changes, and installs the dependencies from `pyproject.toml`. If you see `hello!` printed, everything is working.

## Notes
- **Edit code in the repo, not in Colab.** Commit and push your changes to the files under `assignment/` from your own machine. Re-running the first cell then runs `git pull`, which brings everyone's latest code into Colab.
- **Notebook edits aren't saved automatically.** Any change you make to the notebook itself is lost when the runtime ends unless you use **File → Save a copy in GitHub**.
- **If you get an auth error on clone,** check that you accepted the invite, that the secret name matches exactly, and that Notebook access is on.