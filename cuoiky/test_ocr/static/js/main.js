document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const extractBtn = document.getElementById('extract-btn');
    const resultText = document.getElementById('result-text');
    const loading = document.getElementById('loading');
    const copyBtn = document.getElementById('copy-btn');
    const statsContainer = document.getElementById('stats-container');
    const charCount = document.getElementById('char-count');

    let currentFile = null;

    // Trigger file input
    dropZone.addEventListener('click', () => fileInput.click());

    // Handle drag events
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('drag-over');
        });
    });

    // Handle drop
    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Handle file selection
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Vui lòng chọn file ảnh!');
            return;
        }
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    // Remove image
    removeBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        previewContainer.classList.add('hidden');
        dropZone.classList.remove('hidden');
        resultText.innerHTML = `
            <div class="placeholder-content">
                <i class="fas fa-ghost"></i>
                <p>Chưa có dữ liệu</p>
            </div>
        `;
        copyBtn.classList.add('hidden');
        statsContainer.classList.add('hidden');
    });

    // Extract text
    extractBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        const formData = new FormData();
        formData.append('file', currentFile);

        // UI state
        loading.classList.remove('hidden');
        resultText.classList.add('hidden');
        extractBtn.disabled = true;
        copyBtn.classList.add('hidden');
        statsContainer.classList.add('hidden');

        try {
            const response = await fetch('/ocr', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                resultText.textContent = 'Lỗi: ' + data.error;
            } else {
                resultText.textContent = data.results.join('\n');
                copyBtn.classList.remove('hidden');
                
                // Display character count
                if (data.char_count !== undefined) {
                    charCount.textContent = data.char_count.toLocaleString('vi-VN');
                    statsContainer.classList.remove('hidden');
                }
            }
        } catch (error) {
            resultText.textContent = 'Lỗi kết nối server!';
            console.error(error);
        } finally {
            loading.classList.add('hidden');
            resultText.classList.remove('hidden');
            extractBtn.disabled = false;
        }
    });

    // Copy to clipboard
    copyBtn.addEventListener('click', () => {
        const text = resultText.textContent;
        navigator.clipboard.writeText(text).then(() => {
            const originalIcon = copyBtn.innerHTML;
            copyBtn.innerHTML = '<i class="fas fa-check"></i>';
            copyBtn.style.color = '#55efc4';
            setTimeout(() => {
                copyBtn.innerHTML = originalIcon;
                copyBtn.style.color = '';
            }, 2000);
        });
    });
});
