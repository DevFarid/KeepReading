# KeepReading
OCR Reader for Hard Drives


# Requirements
- Python 3.x+
- Pytesseract
- Tesseract
- pymongo
- Flask
- OpenCV

## Database
For downloads, MongoDB Community version 7.0.2 is required, which can be found here: https://www.mongodb.com/try/download/community 

### Database setup
In the installation wizard, set up a local instance. On Windows, you will need to use your Windows username and password to setup. This is not used for logging into the database, but is used for MongoDB's permissions to setup.

Using command line tools or MongoDB Compass, connect using mongodb://localhost:27017 or a custom connection. Here, you can create a database named keepreading, and a collection named ocrtest (for now, lets change this later)
