# Déploiement Render

Ce projet expose une API FastAPI pour identifier une personne à partir d'une photo.

## Endpoints

- `GET /health`: vérifie que l'API démarre et que le modèle existe.
- `POST /predict`: upload multipart avec le champ `file`.
- `POST /predict-base64`: JSON avec `image_base64`.

Exemple multipart:

```bash
curl -X POST https://TON-SERVICE.onrender.com/predict \
  -F "file=@/chemin/photo.jpg"
```

Exemple base64:

```bash
curl -X POST https://TON-SERVICE.onrender.com/predict-base64 \
  -H "Content-Type: application/json" \
  -d '{"image_base64":"..."}'
```

Réponse typique:

```json
{
  "identity": "kabulu",
  "confidence": 0.91,
  "known": true,
  "scores": {
    "bujiriri": 0.02,
    "kabulu": 0.91,
    "mateo": 0.07
  }
}
```

## Point important: modèle Keras

Render ne verra que les fichiers commités dans Git. Dans ce dépôt, `.gitignore` ignore `models/*.keras`; donc `models/mobilenet_best.keras` ne sera pas déployé sauf si tu l'ajoutes volontairement.

Option simple pour un projet académique:

```bash
git add -f models/mobilenet_best.keras
git add api Dockerfile render.yaml requirements.txt README.deploy.md reports/summary.json
git commit -m "Add Render API for face recognition"
git push
```

Option plus propre pour production: stocker le modèle sur S3, Google Cloud Storage ou autre, puis le télécharger au démarrage du conteneur.

## Déployer sur Render

1. Pousse le projet sur GitHub.
2. Va sur Render, puis `New` > `Blueprint` si tu utilises `render.yaml`, ou `New` > `Web Service`.
3. Connecte ton repo GitHub.
4. Choisis l'environnement Docker.
5. Vérifie les variables:
   - `MODEL_PATH=/app/models/mobilenet_best.keras`
   - `CLASS_NAMES=bujiriri,kabulu,mateo`
   - `UNKNOWN_THRESHOLD=0.60`
6. Déploie, puis teste `/health`.

## Test local

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
curl -X POST http://localhost:8000/predict -F "file=@images/raw/mateo/IMG-20260118-WA0029.jpg"
```
