# Shipping EnviroSense: runbook

Order matters: API first (the site and app both need its URL), then site, then app.

## 0. Security (do this today)
1. Firebase console → Realtime Database → **Rules**: paste the rules from the app's README and publish.
2. In the Data tab, **delete the `systemUser` node** if it exists. It held a plaintext password (`admin123`).
3. Firebase → Authentication → Settings → **Authorized domains**: you'll add your Vercel domains here in step 4.

## 1. Push the three repos to GitHub
```powershell
# ML: move the CI workflows into place (they couldn't be written remotely)
cd C:\Users\hfaisal\Desktop\Repos\Envirosense-ML
mkdir .github\workflows -Force; move deploy\github-workflows\*.yml .github\workflows\
git add -A; git commit -m "v2: real dataset, FastAPI service, reproducible eval"; git push

# Site: wasn't a git repo yet
cd ..\envirosense
git init; git add -A; git commit -m "Live demo wired to model API"
gh repo create envirosense-site --public --source . --push

# App
cd ..\EnviroSense-App\EnviroSense
git add -A; git commit -m "Wire AI tab to model API; env-based config; real alerts"; git push
```
Before pushing the app, run `git status` and make sure `.env` is **not** listed.

## 2. API on Render (free)
1. render.com → New → **Blueprint** → pick the ML repo (it reads `render.yaml`).
2. Wait for the deploy, then open `https://<name>.onrender.com/docs` and try `/predict`.
3. GitHub → ML repo → Settings → Secrets and variables → Actions → **Variables** → add `API_URL` = your Render URL. The keep-warm workflow pings it every 10 min so founders never hit a 30-second cold start.

## 3. Site on Vercel
1. Save the paper as `public/EnviroSense.pdf` in the site repo and push.
2. vercel.com → Add New → Project → import the site repo.
3. Env vars: `NEXT_PUBLIC_API_URL` (Render URL), `NEXT_PUBLIC_GITHUB_ML/SITE/APP` (repo URLs), `NEXT_PUBLIC_CONTACT_EMAIL`.
4. Deploy. Optional: add a custom domain (~$10/yr) under Settings → Domains.

## 4. App on the web (Vercel)
1. Add New → Project → import the app repo.
2. Build command `npx expo export -p web`, output directory `dist`.
3. Env vars: every line from your local `.env`, with `EXPO_PUBLIC_API_URL` set to the Render URL.
4. Deploy, then add the domain to Firebase Authorized domains (step 0.3).
5. Back in the site's Vercel env vars, set `NEXT_PUBLIC_APP_URL` to the app URL and redeploy. The hero button becomes "Open the app".

## 5. Lock CORS (optional, after everything works)
Render → envirosense-api → Environment → `ALLOWED_ORIGINS=https://<site>,https://<app>`.

## 6. Android APK (optional)
```bash
npm i -g eas-cli; eas login; eas build -p android --profile preview
```
Delete the stale `ios/` folder first, or run `npx expo prebuild --clean`; the bundle ID changed.

## Retraining
`python scripts/train.py` → commit `models/` → copy `models/metrics.json` into the site's `lib/metrics.json` → push both.
