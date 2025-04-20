import os
import sys
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler

class CORSHTTPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_GET(self):
        # Special route for keypoints data
        if self.path == '/api/keypoints':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Read keypoints file
            try:
                with open('frontend_dis/data/keypoints.json', 'r') as f:
                    keypoints_data = json.load(f)
                self.wfile.write(json.dumps(keypoints_data).encode('utf-8'))
            except FileNotFoundError:
                self.wfile.write(json.dumps({"error": "Keypoints data not available"}).encode('utf-8'))
            return
        
        # Default handling for all other routes
        return SimpleHTTPRequestHandler.do_GET(self)
        
    def do_OPTIONS(self):
        # Handle preflight requests
        self.send_response(204)
        self.end_headers()

# Change to the frontend_dis directory
os.chdir("frontend_dis")
print(f"Serving files from {os.getcwd()} at http://localhost:8080")

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Start the server
CORSHTTPRequestHandler.extensions_map['.js'] = 'application/javascript'
httpd = HTTPServer(('localhost', 8080), CORSHTTPRequestHandler)

try:
    httpd.serve_forever()
except KeyboardInterrupt:
    print("\nServer stopped by user")
    httpd.server_close()
    sys.exit(0) 