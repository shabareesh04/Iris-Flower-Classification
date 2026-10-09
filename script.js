/**
 * Iris Flower Classification - Client-Side Interaction & Validation
 */

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("prediction-form");
  const submitBtn = document.getElementById("submit-btn");
  const btnText = document.getElementById("btn-text");
  const btnSpinner = document.getElementById("btn-spinner");

  // Sample presets for quick testing
  const presets = {
    setosa: {
      sepal_length: 5.1,
      sepal_width: 3.5,
      petal_length: 1.4,
      petal_width: 0.2,
    },
    versicolor: {
      sepal_length: 5.9,
      sepal_width: 3.0,
      petal_length: 4.2,
      petal_width: 1.5,
    },
    virginica: {
      sepal_length: 6.5,
      sepal_width: 3.0,
      petal_length: 5.5,
      petal_width: 1.8,
    },
  };

  // Wire preset buttons
  document.querySelectorAll(".preset-btn").forEach((button) => {
    button.addEventListener("click", (e) => {
      e.preventDefault();
      const type = button.getAttribute("data-preset");
      const values = presets[type];
      if (values) {
        Object.keys(values).forEach((key) => {
          const input = document.getElementById(key);
          if (input) {
            input.value = values[key];
            clearFieldError(input);
          }
        });
      }
    });
  });

  // Client-side validation boundaries
  const boundaries = {
    sepal_length: { min: 3.0, max: 12.0, label: "Sepal Length" },
    sepal_width: { min: 1.5, max: 6.0, label: "Sepal Width" },
    petal_length: { min: 0.5, max: 10.0, label: "Petal Length" },
    petal_width: { min: 0.1, max: 4.0, label: "Petal Width" },
  };

  function showFieldError(input, message) {
    input.classList.add("input-error");
    const errorEl = document.getElementById(`${input.id}-error`);
    if (errorEl) {
      errorEl.textContent = message;
      errorEl.style.display = "block";
    }
  }

  function clearFieldError(input) {
    input.classList.remove("input-error");
    const errorEl = document.getElementById(`${input.id}-error`);
    if (errorEl) {
      errorEl.textContent = "";
      errorEl.style.display = "none";
    }
  }

  // Real-time input cleaning and error clearing on user typing
  Object.keys(boundaries).forEach((fieldId) => {
    const input = document.getElementById(fieldId);
    if (input) {
      input.addEventListener("input", () => {
        clearFieldError(input);
      });
    }
  });

  // Form submission handler
  if (form) {
    form.addEventListener("submit", (e) => {
      let isValid = true;

      Object.keys(boundaries).forEach((fieldId) => {
        const input = document.getElementById(fieldId);
        if (!input) return;

        const valStr = input.value.trim();
        const config = boundaries[fieldId];

        if (valStr === "") {
          showFieldError(input, `${config.label} is required.`);
          isValid = false;
        } else {
          const num = parseFloat(valStr);
          if (isNaN(num)) {
            showFieldError(input, `${config.label} must be a valid number.`);
            isValid = false;
          } else if (num <= 0) {
            showFieldError(input, `${config.label} must be greater than 0.`);
            isValid = false;
          } else if (num < config.min || num > config.max) {
            showFieldError(
              input,
              `Realistic range: ${config.min} to ${config.max} cm.`
            );
            isValid = false;
          } else {
            clearFieldError(input);
          }
        }
      });

      if (!isValid) {
        e.preventDefault();
        return;
      }

      // Show loading indicator
      if (submitBtn) {
        submitBtn.disabled = true;
        if (btnText) btnText.textContent = "Analyzing Flower...";
        if (btnSpinner) btnSpinner.style.display = "inline-block";
      }
    });
  }

  // Reset button handler
  const resetBtn = document.getElementById("reset-btn");
  if (resetBtn && form) {
    resetBtn.addEventListener("click", () => {
      form.reset();
      Object.keys(boundaries).forEach((fieldId) => {
        const input = document.getElementById(fieldId);
        if (input) clearFieldError(input);
      });
    });
  }
});
