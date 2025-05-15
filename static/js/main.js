// Main JavaScript functionality for the Health Detection System

document.addEventListener('DOMContentLoaded', function() {
    // BMI Calculator
    const heightInput = document.getElementById('height');
    const weightInput = document.getElementById('weight');
    const bmiInput = document.getElementById('bmi');
    const calcBmiBtn = document.getElementById('calcBmi');
    
    if (calcBmiBtn) {
        calcBmiBtn.addEventListener('click', function() {
            calculateBMI();
        });
    }
    
    // Calculate BMI if height and weight are provided
    function calculateBMI() {
        if (heightInput && weightInput && bmiInput) {
            const height = parseFloat(heightInput.value) / 100; // Convert cm to m
            const weight = parseFloat(weightInput.value);
            
            if (height > 0 && weight > 0) {
                const bmi = weight / (height * height);
                bmiInput.value = bmi.toFixed(1);
                
                // Show BMI category
                let category = '';
                if (bmi < 18.5) category = 'Underweight';
                else if (bmi < 25) category = 'Normal weight';
                else if (bmi < 30) category = 'Overweight';
                else category = 'Obese';
                
                // Display BMI category if element exists
                const bmiCategory = document.getElementById('bmiCategory');
                if (bmiCategory) {
                    bmiCategory.textContent = category;
                    bmiCategory.style.display = 'block';
                }
            }
        }
    }
    
    // Form validation with visual feedback
    const healthForm = document.getElementById('health-form');
    
    if (healthForm) {
        const inputs = healthForm.querySelectorAll('input, select');
        
        inputs.forEach(input => {
            input.addEventListener('blur', function() {
                validateInput(this);
            });
        });
        
        healthForm.addEventListener('submit', function(e) {
            let isValid = true;
            
            inputs.forEach(input => {
                if (!validateInput(input)) {
                    isValid = false;
                }
            });
            
            if (!isValid) {
                e.preventDefault();
                showAlert('Please fix the highlighted errors before submitting.', 'danger');
            }
        });
    }
    
    function validateInput(input) {
        // Skip validation for radio buttons and checkboxes
        if (input.type === 'radio' || input.type === 'checkbox') {
            return true;
        }
        
        const value = input.value.trim();
        let isValid = true;
        
        // Clear previous validation
        input.classList.remove('is-invalid');
        input.classList.remove('is-valid');
        
        // Required field validation
        if (input.hasAttribute('required') && value === '') {
            input.classList.add('is-invalid');
            isValid = false;
        }
        
        // Specific validations based on field ID
        if (isValid && input.id) {
            switch (input.id) {
                case 'age':
                    isValid = validateRange(input, 18, 100);
                    break;
                case 'heart_rate':
                    isValid = validateRange(input, 40, 200);
                    break;
                case 'blood_pressure_systolic':
                    isValid = validateRange(input, 80, 200);
                    break;
                case 'blood_pressure_diastolic':
                    isValid = validateRange(input, 40, 130);
                    break;
                case 'cholesterol':
                    isValid = validateRange(input, 100, 300);
                    break;
                case 'blood_sugar':
                    isValid = validateRange(input, 70, 200);
                    break;
                case 'bmi':
                    isValid = validateRange(input, 15, 40);
                    break;
                case 'physical_activity':
                    isValid = validateRange(input, 0, 30);
                    break;
            }
        }
        
        // Add valid class if all checks passed
        if (isValid && value !== '') {
            input.classList.add('is-valid');
        }
        
        return isValid;
    }
    
    function validateRange(input, min, max) {
        const value = parseFloat(input.value);
        
        if (isNaN(value) || value < min || value > max) {
            input.classList.add('is-invalid');
            return false;
        }
        
        return true;
    }
    
    function showAlert(message, type = 'info') {
        // Create alert element
        const alertEl = document.createElement('div');
        alertEl.className = `alert alert-${type} alert-dismissible fade show`;
        alertEl.role = 'alert';
        alertEl.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Find alert container
        const container = document.querySelector('.container');
        
        if (container) {
            // Insert at top of container
            container.insertBefore(alertEl, container.firstChild);
            
            // Auto-dismiss after 5 seconds
            setTimeout(() => {
                alertEl.classList.remove('show');
                setTimeout(() => {
                    alertEl.remove();
                }, 300);
            }, 5000);
        }
    }
    
    // Handle print button
    const printBtn = document.getElementById('printResults');
    if (printBtn) {
        printBtn.addEventListener('click', function() {
            window.print();
        });
    }
});

// Animation for page transitions
window.addEventListener('pageshow', function(event) {
    const main = document.querySelector('main');
    if (main) {
        main.style.opacity = '0';
        setTimeout(() => {
            main.style.transition = 'opacity 0.5s ease';
            main.style.opacity = '1';
        }, 10);
    }
});
