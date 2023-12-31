const { app, BrowserWindow } = require('electron')

let flaskProcess = null;

const createWindow = () => {
    const win = new BrowserWindow({
        width: 1280,
        height: 720,
        minWidth: 640,
        minHeight: 480,
    })

    win.loadURL('http://127.0.0.1:5000')
}

app.whenReady().then(() => {
    createWindow()
})

app.on('window-all-closed', () => {
    app.quit();
});