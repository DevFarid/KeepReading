function setImage() {
    document.getElementById('frame').src = "../static/assets/captured.jpg"
}

const openModal = function () {
    modal.classList.remove("hidden");
    overlay.classList.remove("hidden");
};

const closeModal = function () {
    modal.classList.add("hidden");
    overlay.classList.add("hidden");
}

const loading = function () {
    overlay.classList.remove("hidden");
    loader.classList.remove("hidden");
}

const loadFile = function (event) {
    var image = document.getElementById('upload-image');
    image.src = URL.createObjectURL(event.target.files[0]);
}

