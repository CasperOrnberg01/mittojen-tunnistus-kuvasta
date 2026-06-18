# Instructions to install depedencies for backend and running it

## 1. Locate into backend folder

  ```cd backend```

  ## 2. Create virtual environment (in the backend folder)

  ```python -m venv venv```

  ## 3. Activate virtual environment:

  ```.\venv\Scripts\activate```

  ### Shutdown the virtual environment:

  ``` deactivate ```

  ## 4. Install required depedencies

  ```pip install -r requirements.txt```

  > Required depedencies should automatically be copied from text file "requirements.txt"
  > However, if it doesn't work you can manually install them with the instructions below


  ## Manually installing depedencies (ONLY USE THIS IF YOU SKIPPED PART 4: "Install required depedencies")

  > Use command:
  ```Install dependencies: pip install fastapi uvicorn python-multipart opencv-python numpy```

  ## 5. Start the server:

  ``` uvicorn app.main:app --reload ```

  ### Shutdown the server:

  ``` Ctrl + C ```

## 6. After starting the server, ppen browser and go to http://127.0.0.1:8000/docs

> Here you can test uploading image using POST /upload image
> POST /upload/debug you can use to upload image and in return if A4 found = true U will get JPEG image where A4 paper is surrounded with green outlay + image quality measurements
> POST /upload/warp-debug you can use to return straightened "crop" of the A4 paper if it was found in the image. This feature is a good addition for future development.

