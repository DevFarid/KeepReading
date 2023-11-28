const { app, BrowserWindow } = require('electron')
const { spawn } = require('child_process');

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
    flaskProcess = spawn('python', ['main.py'])
    createWindow()
})

app.on('window-all-closed', () => {
    app.quit();
});

app.on('quit', () => {
    flaskProcess.kill();
});