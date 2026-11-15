# Step-by-step: run the project locally

Everything below is verified to work. The project uses **[uv](https://docs.astral.sh/uv/)**
for environment and dependency management — no `pip`, no manual venv
activation.

Follow the steps in order. At the marked steps, take a screenshot and save it
with the filename given — those are the ones that go into `README.md`.

**Two ways to do this:**

* **Fast path** — the app is already running in this workspace's live preview.
  Look at it right now without installing anything.
* **Local path** — install on your own machine (below). This is what you want
  for screenshots and to prove reproducibility.

---

## What you need

| | |
|---|---|
| Python | **3.12 or newer** (xgboost 3.4.1 requires it) |
| uv | installed once — see Step 0 |
| Disk | ~600 MB (more only if you retrain: the dataset is 471 MB) |
| Docker | **optional** — only for the containerisation screenshot |
| Time | ~5 minutes without retraining, ~15 with |

---

## Part 1 — Get it running

### Step 0 · Install uv (once)

**macOS / Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal afterwards, then confirm:

```bash
uv --version
```

> **Expected:** something like `uv 0.13.8`.
>
> **Already have Python but no curl?** Any of these work too:
> `pip install uv` · `brew install uv` · `winget install astral-sh.uv`

### Step 1 · Unzip

Unzip `fraud-detection.zip` wherever you like:

```
fraud-detection/
├── README.md
├── notebook.ipynb
├── train.py  predict.py  vectorize.py  reasons.py
├── streamlit_app.py  score_and_export.py
├── pyproject.toml  uv.lock  requirements.txt     <- dependency files
├── Dockerfile
├── models/fraud_model.bin          <- the trained model
├── outputs/fraud_report.xlsx       <- the finished report
└── sample_transactions.csv         <- 200-row demo file
```

### Step 2 · Install dependencies

```bash
cd fraud-detection
uv sync
```

That's the whole step. `uv sync` creates `.venv`, reads `uv.lock` and installs
the **exact** versions the reported metrics were produced with.

> **Expected:** a list of resolved packages ending in
> `Installed 53 packages` / `Audited 53 packages`.

### Step 3 · Start the API

Keep this terminal open.

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

> `uv run` executes inside the project environment — **you never need to
> activate the venv**.
>
> **Expected:**
> ```
> INFO:     Application startup complete.
> INFO:     Uvicorn running on http://0.0.0.0:9696
> ```

### Step 4 · Check the API

Open **http://localhost:9696/docs**

You should see Swagger with three endpoints: `/health`, `/predict`, `/score-file`.

> 💡 **Screenshot → save as `01-swagger.png`**
> Make sure the three endpoint names are visible. Bonus: expand `POST /predict`,
> click *Try it out → Execute*, so the response body is in the frame.

Command-line check if you prefer:
```bash
curl http://localhost:9696/health
```
> **Expected:** JSON with `"status": "ok"` and `"pr_auc": 0.9968...`

### Step 5 · Start the web UI

Open a **second** terminal in the same folder.

```bash
uv run streamlit run streamlit_app.py
```

> **Expected:** it prints a Local URL and usually opens your browser to
> **http://localhost:8501**. The sidebar should show a green
> **"backend online"** with PR-AUC 0.9968. If it says *backend offline*, the API
> from Step 3 isn't running.

### Step 6 · Score one transaction (the "fraud" screenshot)

In the **🔍 Single transaction** tab, click **🔴 Fraudulent transfer**, then
**🔍 Score transaction**.

> **Expected:** fraud probability `1.0000`, risk band `critical`, red **🚫 BLOCK**.
>
> 💡 **Screenshot → save as `02-ui-block.png`**

### Step 7 · Score a clean transaction

Click **🟢 Legitimate transfer**, then **🔍 Score transaction**.

> **Expected:** probability `0.0000`, green **✅ ALLOW**.
>
> 💡 **Screenshot → save as `03-ui-allow.png`**

### Step 8 · Upload a file (the batch screenshot)

Switch to the **📂 Batch file** tab.

1. Click **Browse files** and pick `sample_transactions.csv`.
2. Expand **📋 Preview & column mapping** — every column should be recognised
   automatically.
3. Click **🔍 Score file**.

> **Expected:** `rows scored 200`, `fraud found 38`, `flagged rate 19.00%`.
> The sample file has ground-truth labels, so a green box also reports
> **precision 1.000, recall 0.950, PR-AUC 0.980** on your own data.
>
> 💡 **Screenshot → save as `04-batch-results.png`**
> Capture the four metric tiles *and* a few table rows, including
> **why_flagged**.

### Step 9 · Download the report

Still on the batch tab, click **⬇️ Download Excel**, open it, and screenshot the
**Results** sheet.

> 💡 **Screenshot → save as `05-batch-excel.png`**

Also open the bundled `outputs/fraud_report.xlsx` → **Suspected Fraud** sheet,
showing `dt_reason` / `if_reason`.

> 💡 **Screenshot → save as `06-full-report.png`**

### Step 10 · Docker (optional but worth 2 rubric points)

In a third terminal, from the project folder:

```bash
# Behind a proxy? Pass it in — HTTP_PROXY / HTTPS_PROXY are Docker's PREDEFINED
# build args, so no Dockerfile change is needed and nothing is baked into the image.
docker build \
  --network=host \
  --build-arg HTTP_PROXY="$HTTP_PROXY" \
  --build-arg HTTPS_PROXY="$HTTPS_PROXY" \
  --build-arg http_proxy="$http_proxy" \
  --build-arg https_proxy="$https_proxy" \
  --build-arg NO_PROXY="$NO_PROXY" \
  -t fraud-detection .

# No proxy? Use the plain build:
docker build -t fraud-detection .

docker run --rm -p 9696:9696 -v "$(pwd)/models:/app/models" fraud-detection
```

**Windows PowerShell** — use `${PWD}` instead of `$(pwd)`:
```powershell
docker run --rm -p 9696:9696 -v "${PWD}/models:/app/models" fraud-detection
```

> **Expected:** `Successfully built ...`, then the same uvicorn startup lines.
> **Stop the Step 3 server first**, or port 9696 will already be in use.
> (The image installs with uv too — Python 3.12 base.)

> **If the build fails with `error resolving DNS`:** your shell probably exports
> a proxy that the build container doesn't inherit. Check with
> `env | grep -i proxy` — if it prints anything, use the proxy `docker build`
> command above. That is the fix that worked on the reference machine.
> Otherwise set the daemon's DNS once (`/etc/docker/daemon.json`, then
> `sudo systemctl restart docker`):
> ```json
> { "dns": ["8.8.8.8", "8.8.4.4", "1.1.1.1"] }
> ```

> 💡 **Screenshot → save as `07-docker.png`**
> Capture the terminal showing the build finishing *and* the container running.

### Step 10b · Run the UI in Docker too (optional)

`Dockerfile` contains the API only — the UI is a separate image on purpose, so
the two can be scaled independently. To bring up both:

```bash
sudo -E docker compose up --build     # -E is REQUIRED if you use sudo
```

`sudo` strips your environment by default, which leaves the proxy variables
empty and the build fails with `error resolving DNS`. `-E` preserves them.
Better still, drop `sudo` for Docker once:

```bash
sudo usermod -aG docker $USER    # then log out and back in
docker compose up --build
```

* **API** → http://localhost:9696/docs
* **UI** → http://localhost:8501

The compose file points the UI at `http://api:9696` and waits for the API's
healthcheck before starting it.

> 💡 **Screenshot → save as `08-docker-compose.png`**
> Capture both containers running, or the UI loading in the browser at 8501.

---

## Part 2 — Running the notebook

`notebook.ipynb` must run **inside the project environment**. If you open it
with a globally-installed Jupyter it uses the system pyarrow instead of the
pinned one and crashes with errors like
`module 'pyarrow.compute' has no attribute 'modulo'`.

```bash
uv run --with jupyter jupyter lab
```

Then open `notebook.ipynb` from the browser tab that appears. To install Jupyter
into the project permanently instead: `uv add --dev jupyterlab`.

## Part 3 — Deploy to Streamlit Community Cloud (optional, +2 rubric points)

The free tier needs a **public** GitHub repo.

### Step 1 · Push the project

#### 1a · Create an empty repo on GitHub

**https://github.com/new** → name `fraud-detection` → **Public** → leave
*Add a README file*, *Add .gitignore* and *Choose a license* **all unticked**.

An empty repo avoids the "unrelated histories" rejection on the first push, and
stops GitHub's generic `.gitignore` from clobbering the project's own one.

#### 1b · Set up SSH (one time)

GitHub has not accepted account passwords over HTTPS since 2021, so pushing
over HTTPS fails with:

```
remote: Invalid username or token. Password authentication is not supported for Git operations.
```

```bash
ssh-keygen -t ed25519 -C "your-github-username"     # press Enter 3x
eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

Copy that output → **https://github.com/settings/keys** → **New SSH key** →
paste → save. Then:

```bash
ssh -T git@github.com
# expect: "Hi <username>! You've successfully authenticated"
```

**Prefer a token?** **https://github.com/settings/tokens** → *Fine-grained* →
repository access `fraud-detection` → permissions **Contents: Read and write**.
When git asks for a password, paste the **token**. Or `sudo apt install gh &&
gh auth login` and let it store the credential for you.

#### 1c · Point the project at the repo

```bash
cd ~/Downloads/fraud-detection
git init                     # skip if it is already a git repo
git remote add origin git@github.com:<you>/fraud-detection.git
```

Already cloned over HTTPS? Switch it:

```bash
git remote set-url origin git@github.com:<you>/fraud-detection.git
```

#### 1d · Commit — the model must be in it

```bash
git add -A
git add -f models/fraud_model.bin     # -f overrides any stale ignore rule
git status --short                    # models/fraud_model.bin MUST appear
git commit -m "PaySim fraud detection: CRISP-DM pipeline, FastAPI + Streamlit, Docker, cloud-ready"
```

If any `data/` or `outputs/` line appears in `git status`, fix `.gitignore`
before committing.

#### 1e · Push

```bash
git push -u origin main
```

> **Created the repo with a README/.gitignore/licence ticked?** The first push
> is rejected as unrelated histories. Force-push is the clean fix — and *only*
> in this case, because the remote holds nothing but GitHub's template files:
>
> ```bash
> git push -f origin main
> ```
>
> `git pull --rebase` also works but overwrites the project `.gitignore` with
> GitHub's generic one, so prefer the force-push here.

#### 1f · Confirm

```bash
git ls-remote --heads origin
```

Refresh the repo page: ~44 files, including `inference.py`,
`requirements-cloud.txt` and `models/fraud_model.bin`.

`data/` and `outputs/` are gitignored so the push stays small.
`models/fraud_model.bin` is committed on purpose (3.7 KB) — the cloud app needs
it to score in-process.

### Step 2 · Create the app

1. **https://share.streamlit.io** → sign in with GitHub → **New app**
2. Repo and branch → **Main file path**: `streamlit_app.py`
3. **Advanced settings** (open this before deploying):
   * **Python version**: `3.12`
   * **Requirements file**: `requirements-cloud.txt`
4. **Deploy**

### Step 3 · Verify

The sidebar should show the amber **"no backend — running the model in-
process"** message. That is correct in the cloud — Streamlit Cloud gives one
container, so there is no API to call and the app loads the model itself.

* **🔍 Single transaction** → 🔴 Fraudulent transfer → 🚫 BLOCK
* **📂 Batch file** → upload `sample_transactions.csv` → 38 fraud found

> 💡 **Screenshot → save as `09-streamlit-cloud.png`**
> The deployed app in the browser at its public `*.streamlit.app` URL.

## Part 4 — Optional: retrain from scratch

Only needed to prove the model reproduces, or if you want to change it. This
downloads the 471 MB PaySim dataset.

```bash
mkdir -p data
curl -L -o data/paysim.csv https://huggingface.co/datasets/vitaliy-sharandin/synthetic-fraud-detection/resolve/main/PS_20174392719_1491204439457_log.csv

uv run python train.py                 # ~2-4 min, ~4 GB RAM
uv run python score_and_export.py      # ~90 s -> outputs/fraud_report.xlsx
```

> **Expected from `train.py`:** `TEST PR-AUC = 0.9968`, `TEST ROC-AUC = 0.9992`.
> If it runs out of memory: `PAYSIM_FIT_SPLIT=train uv run python train.py`.

---

---

## Part 5 — Troubleshooting


| Symptom | Fix |
|---|---|
| `uv: command not found` | Re-run Step 0 and restart your terminal |
| `requires Python >=3.12` | Install Python 3.12+ (xgboost 3.4.1 needs it) |
| `backend offline` in the sidebar | The API isn't running — do Step 3 |
| `address already in use` | Stop the other server, or run on `--port 9697` and start the UI with `API_URL=http://localhost:9697` |
| UI says "Missing required column(s)" | Map `amount` / `oldbalanceOrg` by hand in the Preview & column mapping panel |
| Docker: `model not found` | You forgot the `-v .../models:/app/models` mount |
| Docker build: `error resolving DNS` | Build with `docker build --network=host -t fraud-detection .` (see below) |
| Want to start clean | Delete `.venv` and re-run `uv sync` |

**Prefer pip?** `requirements.txt` is still shipped, so this also works — it
just won't be pinned to the lockfile:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Part 6 — Send me the screenshots

Attach the PNGs to your next message (any filenames work — ideally the ones
above). I'll then:

1. Save them into `midterm/images/`
2. Add a **Screenshots** section to `README.md` with each image embedded
3. Rebuild `fraud-detection.zip` so everything stays in one file

**Minimum for the rubric:** `01-swagger` plus either `02-ui-block` or
`03-ui-allow` — those prove deployment. The rest make the README stronger.
