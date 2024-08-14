document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const signInButton = document.getElementById('signInButton');
    const signOutButton = document.getElementById('signOutButton');
    const signInForm = document.getElementById('signInForm');
    const form = document.getElementById('form');
    const restaurantNameElement = document.getElementById('restaurantName');
    const addFridgeButton = document.getElementById('addFridgeButton');
    const fridgeContainer = document.getElementById('fridgeContainer');
    const uploadForm = document.getElementById('uploadForm');
    const translateButton = document.getElementById('translate-btn');

    // Initial state
    let isSignedIn = false; // This should be updated based on your actual sign-in status
    let fridges = [];
    const translations = {
        en: {
            title: "Basira",
            inventory: "Inventory",
            forecasting: "Inventory Forecasting",
            items: "Items in your fridge:",
            copyright: "© 2024 Basira App. All rights reserved",
            stockPrediction: "Stock Prediction",
            homePage: "Home Page",
        },
        ar: {
            title: "بصيرة",
            inventory: "المخزون",
            forecasting: "التنبؤ بالمخزون",
            items: "يوجد في ثلاجتك :",
            copyright: "© 2024 تطبيق بصيرة.كل الحقوق محفوظة",
            stockPrediction: "التنبؤ بالمخزون",
            homePage: "الصفحة الرئيسية",
        }
    };
    let currentLang = 'ar'; // Default language

    // Update button state based on sign-in status
    function updateButtonState() {
        if (isSignedIn) {
            signInButton.classList.add('d-none');
            signOutButton.classList.remove('d-none');
        } else {
            signInButton.classList.remove('d-none');
            signOutButton.classList.add('d-none');
        }
    }

    // Show/hide sign-in form
    signInButton.addEventListener('click', function() {
        signInForm.classList.toggle('d-none');
    });

    // Handle form submission
    form.addEventListener('submit', function(event) {
        event.preventDefault();
        const restaurantName = document.getElementById('restaurantNameInput').value;
        restaurantNameElement.textContent = restaurantName;
        signInForm.classList.add('d-none');
        isSignedIn = true;
        updateButtonState();
    });

    // Handle sign-out
    signOutButton.addEventListener('click', function() {
        isSignedIn = false;
        updateButtonState();
    });

    // Handle fridge addition
    addFridgeButton.addEventListener('click', function() {
        const id = Date.now();
        const name = `Fridge ${fridges.length + 1}`;
        fridges.push({ id, name });

        const fridgeDiv = document.createElement('div');
        fridgeDiv.className = 'bg-white shadow-md rounded-lg p-4 text-center m-2 card';
        fridgeDiv.innerHTML = `
            <i class="fas fa-refrigerator fa-5x mb-4 text-primary"></i>
            <p class="h6 font-weight-bold">${name}</p>
        `;
        fridgeContainer.appendChild(fridgeDiv);
    });

    // Handle forecasting form submission
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(event) {
            event.preventDefault();
            const formData = new FormData(this);

            fetch('/upload', {
                method: 'POST',
                body: formData
            }).then(response => response.text())
              .then(data => {
                document.body.innerHTML = data;
              })
              .catch(error => {
                console.error('Error:', error);
              });
        });
    }

    // Handle language translation
    translateButton.addEventListener('click', function() {
        currentLang = (currentLang === 'en') ? 'ar' : 'en';
        translateButton.textContent = currentLang === 'en' ? 'AR' : 'EN';

        document.title = translations[currentLang].title;
        document.querySelector('h1').textContent = translations[currentLang].title;
        document.querySelector('h2').textContent = translations[currentLang].inventory;
        document.querySelector('p').textContent = translations[currentLang].items;
        document.querySelector('footer p').textContent = translations[currentLang].copyright;
        document.querySelector('.btn-light span').textContent = translations[currentLang].stockPrediction;
        document.querySelectorAll('.btn-primary span').forEach(btn => {
            btn.textContent = translations[currentLang].homePage;
        });
    });

    // Initialize button state
    updateButtonState();
});
