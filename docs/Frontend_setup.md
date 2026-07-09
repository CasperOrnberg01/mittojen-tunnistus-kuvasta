# Instructions to install dependencies for frontend and run it

## 1. Navigate into the frontend folder
```bash
cd frontend
```

## 2. Install required dependencies
All frontend dependencies are listed in `package.json`.

Install them using:
```bash
npm install
```

This will automatically install:
- React
- React Router
- Vite
- All additional libraries required by the project

## 3. Start the development server
```bash
npm run dev
```

After starting, the frontend will be available at:
```
http://localhost:5173
```

Open this URL in your browser to access the application.

## 4. Build the production version (optional)
```bash
npm run build
```

The optimized production build will be generated in:
```
dist/
```

## 5. Preview the production build (optional)
```bash
npm run preview
```

This allows you to test the production build locally.

## 6. Connecting frontend to backend
The frontend communicates with the backend using the endpoint:
```
POST http://127.0.0.1:8000/upload
```

To ensure proper communication:
- The backend server must be running
- CORS must be enabled on the backend
- The frontend must be running on `http://localhost:5173`

If the backend is not running, the frontend will show:
```
Failed to fetch
```

If CORS is not configured, you will see:
```
Blocked by CORS policy
```

## 7. Testing the connection
After both servers are running:

1. Open the frontend:
   ```
   http://localhost:5173
   ```

2. Upload an image using the Upload page.

3. Open browser DevTools → Console  
   You should see:
   ```
   Sending file to backend...
   Backend response: {...}
   ```

4. Open DevTools → Network → select the `upload` request  
   - Status should be **200 OK**  
   - Response should contain JSON with analysis data

## 8. Shutting down the frontend
To stop the development server:
```
Ctrl + C
```
