# Portainer migration: local image → GHCR image

Apply only after the image has been published and verified. Nothing here touches Paperless, PostgreSQL, Redis, or any data volume.

## 1. Verify the published image has the corrected script

```bash
docker pull ghcr.io/mrjesseking/paperless-preprocessor:latest
docker run --rm --entrypoint grep ghcr.io/mrjesseking/paperless-preprocessor:latest \
  -c "OllamaAnalysisError" /app/processor.py     # expect a number > 0
docker run --rm --entrypoint grep ghcr.io/mrjesseking/paperless-preprocessor:latest \
  "PREPROCESSOR VERSION" /app/processor.py       # expect 2026-11-11a
```

## 2. Edit the `preprocessor` service

Change `image:` and remove **only** the script bind mount. Keep everything else (service name `preprocessor`, container name `paperless-preprocessor`, `depends_on: webserver`, the incoming/consume/work mounts, environment values).

```yaml
  preprocessor:
-   image: paperless-preprocessor:test
+   image: ghcr.io/mrjesseking/paperless-preprocessor:latest
    container_name: paperless-preprocessor
    depends_on:
      - webserver
    volumes:
      # keep your existing incoming / consume / work mounts unchanged
-     - /opt/paperless-preprocessor/processor.py:/app/processor.py:ro
```

Why: a bind mount at `/app/processor.py` overrides the script baked into the image, so the new image would silently keep running the old host file. ([Docker bind mounts](https://docs.docker.com/engine/storage/bind-mounts/))

Then in Portainer: **Update the stack** (enable "Re-pull image" so the new image is fetched).

## 3. Rollback

Keep the old local image `paperless-preprocessor:test` and `/opt/paperless-preprocessor/processor.py` on the host. To roll back, set `image: paperless-preprocessor:test` and **restore** `- /opt/paperless-preprocessor/processor.py:/app/processor.py:ro` (the old image needs its corrected script mounted). Do not remove database/data volumes or change Paperless/PostgreSQL/Redis versions.

## 4. Back up the local image before any pruning

```bash
docker save paperless-preprocessor:test -o ~/paperless-preprocessor-test.tar
```

Do not run `docker image prune` until the GHCR image is confirmed working and you have this backup.

## 5. Later updates

- `:latest` follows `main`. Redeploy or "Re-pull image" in Portainer to update.
- Pin for reproducibility: `ghcr.io/mrjesseking/paperless-preprocessor:0.1.0` (a `v0.1.0` tag), or `:sha-<short>`, or a digest `@sha256:...`.
