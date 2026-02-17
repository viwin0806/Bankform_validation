/**
 * BankForm-AI Frontend Application
 * Main application logic for form processing
 */

// Configuration
const CONFIG = {
    API_URL: 'http://localhost:5050',
    MAX_FILE_SIZE: 16 * 1024 * 1024, // 16MB
    ALLOWED_EXTENSIONS: ['png', 'jpg', 'jpeg', 'pdf', 'tiff']
};

// Application State
const state = {
    currentFile: null,
    currentFilePath: null,
    selectedTemplate: 'deposit_slip',
    useTemplate: false,
    processedData: null,
    templates: []
};

// DOM Elements
const elements = {
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    uploadPreview: document.getElementById('uploadPreview'),
    previewImage: document.getElementById('previewImage'),
    removeBtn: document.getElementById('removeBtn'),
    processBtn: document.getElementById('processBtn'),
    useTemplateToggle: document.getElementById('useTemplate'),
    templateGrid: document.getElementById('templateGrid'),
    processingStatus: document.getElementById('processingStatus'),
    resultsSection: document.getElementById('resultsSection'),
    resultsGrid: document.getElementById('resultsGrid'),
    confidenceBadge: document.getElementById('confidenceBadge'),
    confidenceText: document.getElementById('confidenceText'),
    confidenceValue: document.getElementById('confidenceValue'),
    exportCsvBtn: document.getElementById('exportCsvBtn'),
    exportJsonBtn: document.getElementById('exportJsonBtn'),
    exportExcelBtn: document.getElementById('exportExcelBtn'),
    processAnotherBtn: document.getElementById('processAnotherBtn'),
    statsGrid: document.getElementById('statsGrid'),
    statTotal: document.getElementById('statTotal'),
    statConfidence: document.getElementById('statConfidence'),
    statApproved: document.getElementById('statApproved')
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    loadTemplates();
    loadStatistics();

    // Refresh stats every 30 seconds
    setInterval(loadStatistics, 30000);
});

// Event Listeners
function initializeEventListeners() {
    // Upload area events
    elements.uploadArea.addEventListener('click', () => {
        elements.fileInput.click();
    });

    elements.fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    elements.uploadArea.addEventListener('dragover', handleDragOver);
    elements.uploadArea.addEventListener('dragleave', handleDragLeave);
    elements.uploadArea.addEventListener('drop', handleDrop);

    // Remove file
    elements.removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    // Process button
    elements.processBtn.addEventListener('click', processForm);

    // Template toggle
    elements.useTemplateToggle.addEventListener('change', (e) => {
        state.useTemplate = e.target.checked;
    });

    // Export buttons
    elements.exportCsvBtn.addEventListener('click', () => exportData('csv'));
    elements.exportJsonBtn.addEventListener('click', () => exportData('json'));
    elements.exportExcelBtn.addEventListener('click', () => exportData('excel'));

    // Process another
    elements.processAnotherBtn.addEventListener('click', () => {
        resetUpload();
        elements.resultsSection.classList.add('hidden');
    });
}

// Load Available Templates
async function loadTemplates() {
    try {
        const response = await fetch(`${CONFIG.API_URL}/api/templates`);
        const data = await response.json();

        state.templates = data.templates || [];
        renderTemplates();
    } catch (error) {
        console.error('Error loading templates:', error);
        showNotification('Could not load templates', 'error');
    }
}

function renderTemplates() {
    elements.templateGrid.innerHTML = '';

    state.templates.forEach(template => {
        const card = document.createElement('div');
        card.className = `template-card ${template.id === state.selectedTemplate ? 'active' : ''}`;
        card.innerHTML = `
            <h4>${template.name}</h4>
            <p>${template.bank_name}</p>
        `;
        card.addEventListener('click', () => selectTemplate(template.id));
        elements.templateGrid.appendChild(card);
    });
}

function selectTemplate(templateId) {
    state.selectedTemplate = templateId;
    renderTemplates();
}

// File Handling
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        validateAndDisplayFile(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    elements.uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('dragover');

    const file = e.dataTransfer.files[0];
    if (file) {
        validateAndDisplayFile(file);
    }
}

function validateAndDisplayFile(file) {
    // Check file type
    const extension = file.name.split('.').pop().toLowerCase();
    if (!CONFIG.ALLOWED_EXTENSIONS.includes(extension)) {
        showNotification('Invalid file type. Please upload PNG, JPG, JPEG, PDF, or TIFF files.', 'error');
        return;
    }

    // Check file size
    if (file.size > CONFIG.MAX_FILE_SIZE) {
        showNotification('File too large. Maximum size is 16MB.', 'error');
        return;
    }

    // Display preview
    state.currentFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        elements.previewImage.src = e.target.result;
        elements.uploadArea.classList.add('hidden');
        elements.uploadPreview.classList.remove('hidden');
        elements.processBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    state.currentFile = null;
    state.currentFilePath = null;
    elements.fileInput.value = '';
    elements.previewImage.src = '';
    elements.uploadArea.classList.remove('hidden');
    elements.uploadPreview.classList.add('hidden');
    elements.processBtn.disabled = true;
}

// Form Processing
async function processForm() {
    if (!state.currentFile) return;

    try {
        // Show processing status
        elements.processingStatus.classList.remove('hidden');
        elements.resultsSection.classList.add('hidden');
        elements.processBtn.disabled = true;

        // Upload file
        const formData = new FormData();
        formData.append('file', state.currentFile);

        const uploadResponse = await fetch(`${CONFIG.API_URL}/api/upload`, {
            method: 'POST',
            body: formData
        });

        const uploadData = await uploadResponse.json();

        if (!uploadData.success) {
            throw new Error(uploadData.error || 'Upload failed');
        }

        state.currentFilePath = uploadData.filepath;

        // Process form
        const processResponse = await fetch(`${CONFIG.API_URL}/api/process`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filepath: state.currentFilePath,
                form_type: state.selectedTemplate,
                use_template: state.useTemplate
            })
        });

        const processData = await processResponse.json();

        if (processData.error) {
            throw new Error(processData.error);
        }

        // Display results
        state.processedData = processData;
        displayResults(processData);

        // Update statistics
        loadStatistics();

    } catch (error) {
        console.error('Processing error:', error);
        showNotification(`Error: ${error.message}`, 'error');
    } finally {
        elements.processingStatus.classList.add('hidden');
        elements.processBtn.disabled = false;
    }
}

function displayResults(data) {
    // Update confidence badge
    const confidence = data.overall_confidence || 0;
    const confidencePercent = Math.round(confidence * 100);

    elements.confidenceValue.textContent = `${confidencePercent}%`;
    elements.confidenceText.textContent = data.confidence_flag?.message || 'N/A';

    // Set badge color
    elements.confidenceBadge.classList.remove('green', 'yellow', 'red');
    const flag = data.confidence_flag?.flag || 'green';
    elements.confidenceBadge.classList.add(flag);

    // Render fields
    elements.resultsGrid.innerHTML = '';

    data.fields.forEach(field => {
        const fieldCard = document.createElement('div');
        fieldCard.className = `result-field ${field.is_valid ? '' : 'invalid'}`;

        const confidencePercent = Math.round((field.confidence || 0) * 100);

        fieldCard.innerHTML = `
            <div class="field-header">
                <span class="field-name">${formatFieldName(field.field_name)}</span>
                <span class="field-confidence">${confidencePercent}%</span>
            </div>
            <div class="field-value">${field.corrected_value || field.extracted_value || 'N/A'}</div>
            ${!field.is_valid ? `<div class="field-validation">⚠️ ${field.validation_message}</div>` : ''}
        `;

        elements.resultsGrid.appendChild(fieldCard);
    });

    // Show results section
    elements.resultsSection.classList.remove('hidden');

    // Smooth scroll to results
    elements.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function formatFieldName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

// Export Functionality
async function exportData(format) {
    try {
        const response = await fetch(`${CONFIG.API_URL}/api/export`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                format: format,
                status: 'approved' // or null for all
            })
        });

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `bankform_export.${format === 'excel' ? 'xlsx' : format}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        showNotification(`Exported successfully as ${format.toUpperCase()}`, 'success');
    } catch (error) {
        console.error('Export error:', error);
        showNotification('Export failed', 'error');
    }
}

// Statistics
async function loadStatistics() {
    try {
        const response = await fetch(`${CONFIG.API_URL}/api/stats`);
        const stats = await response.json();

        elements.statTotal.textContent = stats.total_forms || 0;
        elements.statConfidence.textContent = stats.average_confidence
            ? `${Math.round(stats.average_confidence * 100)}%`
            : '0%';
        elements.statApproved.textContent = stats.auto_approved || 0;
    } catch (error) {
        console.error('Error loading statistics:', error);
    }
}

// Notifications
function showNotification(message, type = 'info') {
    // Simple notification (can be enhanced with a toast library)
    console.log(`[${type.toUpperCase()}] ${message}`);
    alert(message);
}

// Utility Functions
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR'
    }).format(amount);
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-IN');
}

console.log('BankForm-AI Application Initialized');
