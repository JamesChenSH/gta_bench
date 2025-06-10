# llama_cpp_debug_server.py
import logging
import sys
import argparse
import json
import os
from fastapi import Request
from llama_cpp.server.app import create_app

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('server_debug.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Path to config file")
    args = parser.parse_args()
    
    if not args.config_file:
        logger.error("Config file is required")
        sys.exit(1)
    
    # Set environment variable for config file
    os.environ['CONFIG_FILE'] = args.config_file
    
    # Create app without explicit settings, let it read from environment
    app = create_app()

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        # Log request details
        try:
            body = await request.body()
            logger.info(f"=== REQUEST ===")
            logger.info(f"Method: {request.method}")
            logger.info(f"URL: {request.url}")
            logger.info(f"Headers: {dict(request.headers)}")
            if body:
                try:
                    body_str = body.decode('utf-8')
                    logger.info(f"Body: {body_str}")
                except:
                    logger.info(f"Body (bytes): {body}")
            logger.info(f"=== END REQUEST ===")
        except Exception as e:
            logger.error(f"Error logging request: {e}")
        
        response = await call_next(request)
        return response

    # Load config to get host/port
    with open(args.config_file, 'r') as f:
        config_data = json.load(f)
    
    host = config_data.get('host', '0.0.0.0')
    port = config_data.get('port', 8000)
    
    # Run the server
    import uvicorn
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="debug")

if __name__ == "__main__":
    main()
