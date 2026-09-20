#!/usr/bin/env bash
# Two ways to use the image: execute the notebook once, or serve it.
set -euo pipefail

case "${1:-run}" in
  run)
    # Execute end to end and write the results back into the mounted output/.
    echo "Executing sms_pipeline.ipynb ..."
    exec jupyter nbconvert --to notebook --execute --inplace \
         --ExecutePreprocessor.timeout=7200 sms_pipeline.ipynb
    ;;
  notebook|lab)
    # Serve the notebook. No token, because the port should not be published
    # beyond localhost; bind it with -p 127.0.0.1:8888:8888.
    exec jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser \
         --IdentityProvider.token='' --ServerApp.password=''
    ;;
  shell)
    exec /bin/bash
    ;;
  *)
    exec "$@"
    ;;
esac
