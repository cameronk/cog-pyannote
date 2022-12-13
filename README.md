# cog-pyannote

## Useful commands

```
nvidia-smi
htop
docker exec sd cat predict.log
docker run --publish 80:5000 --name sd cog-speaker-diarization
docker builder prune

curl -X POST -H "Content-Type: application/json" -d '{"input": {"audio": "https://github.com/cameronk/cog-pyannote/blob/diarize/speaker-diarization/examples/jre-kevin-hart-youtube-short-vzx6h2sAGTU.wav?raw=true", "auth_token": "TOKEN_HERE"}}' http://127.0.0.1/predictions
```

## VM setup

1. https://docs.docker.com/desktop/install/debian/
2. https://replicate.com/docs/guides/push-a-model#install-cog