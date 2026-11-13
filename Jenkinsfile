pipeline {
  agent any
  environment { IMAGE = 'cad-detector'; PORT = '5000'; MODELS_DIR = '/opt/cad/models' }
  options { timestamps(); disableConcurrentBuilds() }
  stages {
    stage('Checkout') { steps { checkout scm } }
    stage('Unit Tests') {
      steps { sh '''
        python3 -m venv .venv
        . .venv/bin/activate
        pip install -q flask pillow numpy pytest
        pytest -q
      ''' }
    }
    stage('Docker Build') { steps { sh 'docker build -t $IMAGE:$BUILD_NUMBER -t $IMAGE:latest .' } }
    stage('Deploy') {
      steps { sh '''
        mkdir -p $MODELS_DIR
        docker rm -f cad-app || true
        docker run -d --name cad-app --restart unless-stopped -p $PORT:5000 -v $MODELS_DIR:/app/models:ro $IMAGE:latest
        for i in $(seq 1 30); do curl -fs http://localhost:$PORT/health && exit 0; sleep 4; done
        docker logs cad-app; exit 1
      ''' }
    }
  }
  post { always { sh 'docker image prune -f || true' } }
}
