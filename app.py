from fastapi import FastAPI

app = FastAPI(title='Vision AI')


@app.get('/')
def root():
    return {
        'service': 'vision-ai',
        'status': 'running'
    }


@app.get('/health')
def health():
    return {
        'ok': True,
        'service': 'vision-ai'
    }
