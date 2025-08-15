# Streamlit Community Cloud Deployment Guide

This guide walks you through deploying the Customer Support RAG System to Streamlit Community Cloud with secure secrets handling.

## 1. Prerequisites
- GitHub account & this repository pushed (public or private – private requires Cloud plan)
- OpenAI API key
- Pinecone API key (Serverless project, region us-east-1 recommended)

## 2. Repository Layout Requirements
Already satisfied:
- `app.py` at project root (Streamlit entrypoint)
- `requirements.txt` listing dependencies
- `.streamlit/config.toml` present (optional UI/theme)
- `config.py` reads from environment or `st.secrets`

## 3. Push to GitHub
```bash
# If not yet a git repo
# git init
# git add .
# git commit -m "Initial commit"
# git remote add origin https://github.com/<you>/<repo>.git
# git push -u origin main
```

## 4. Create the Streamlit App
1. Go to https://share.streamlit.io
2. Click: New app
3. Select repo, branch: `main`, file path: `app.py`
4. Deploy (it will fail if secrets missing – add them next)

## 5. Add Secrets (Settings → Secrets)
Paste a TOML block (NOT JSON):
```toml
OPENAI_API_KEY = "sk-..."
PINECONE_API_KEY = "pc-..."
PINECONE_ENVIRONMENT = "us-east-1"
PINECONE_INDEX_NAME = "customer-support-rag"
# Optional overrides
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
ESCALATION_THRESHOLD = "0.3"
HIGH_PRIORITY_THRESHOLD = "0.2"
MAX_CONTEXT_LENGTH = "4000"
MAX_RESPONSE_LENGTH = "500"
TEMPERATURE = "0.7"
# Faster cold start (optional: skip index init until first search)
FAST_INIT = "1"
```
Save → triggers a redeploy.

### Why TOML strings?
Streamlit secrets parse everything as TOML; quoting maintains exact values. `config.py` casts numeric values where needed.

## 6. First Run Notes
- If the Pinecone index does not exist, the app creates it (≈15s delay). A status message appears.
- If you see warnings about missing secrets, confirm they are in Settings → Secrets & redeploy.

## 7. Updating Secrets
Edit under Settings → Secrets. Each save triggers an automatic redeploy.

## 8. Pinecone Serverless Setup (If New)
In Pinecone console:
1. Create API key (region us-east-1)
2. Ensure project type is Serverless compatible
3. No manual index creation needed (app creates index automatically)

## 9. Local Dev vs Cloud
Local: `.env` file + `python-dotenv`
Cloud: `st.secrets` (do NOT commit `.env`)
`config.py` checks: ENV VAR → `st.secrets` → default.

## 10. Common Issues
| Symptom | Cause | Fix |
|---------|-------|-----|
| "Failed to initialize Pinecone index" | Wrong key or region | Verify keys & region in secrets |
| Empty retrieval results | Index still empty | Use sidebar Reload Knowledge Base |
| Long cold start | Index creation & caching | Wait; subsequent runs are faster |
| Rate limit errors | OpenAI usage spikes | Add backoff or upgrade plan |

## 11. Optional Hardening
- Add simple auth (e.g., share passphrase) via `st.text_input` at session start.
- Limit message count per session (`st.session_state.message_counter`).
- Add usage logging (without storing PII) to a lightweight backend.

## 12. Manual Redeploy
In Streamlit Cloud app page → Rerun or Clear cache if you have updated dependencies or logic.

## 13. Dependency Changes
Modify `requirements.txt` → Commit → Push → Auto redeploy. If a library fails to build, pin a known compatible version.

## 14. Testing Before Deploy
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py
```

## 15. Verifying Secrets in App (Temporary Debug)
Add in a local branch (DO NOT COMMIT KEYS):
```python
import streamlit as st, config
st.write(bool(config.OPENAI_API_KEY), bool(config.PINECONE_API_KEY))
```
Remove before pushing.

---
Deployment complete! Your app is reachable at:
`https://<your-username>-<repo-name>-<random>.streamlit.app`
