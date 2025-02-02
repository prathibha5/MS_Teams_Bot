const chatWindow = document.getElementById('chat-window');
const queryForm = document.getElementById('query-form');

function addMessage(message, sender) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', sender);
    messageElement.innerText = message;
    chatWindow.appendChild(messageElement);
}

async function sendMessage(query, sourceLang) {
    try {
        const response = await fetch('http://127.0.0.1:5000/classify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                source_lang: sourceLang
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();
        addMessage(`Predicted Label: ${result.predicted_label}`, 'received');
    } catch (error) {
        console.error('Error:', error);
        addMessage(`Error: ${error.message}`, 'received');
    }
}

queryForm.addEventListener('submit', function(event) {
    event.preventDefault();
    const sourceLang = document.getElementById('source-lang').value;
    const query = document.getElementById('query').value;

    addMessage(`Source Language: ${sourceLang}`, 'sent');
    addMessage(`Query: ${query}`, 'sent');
    sendMessage(query, sourceLang);
});
