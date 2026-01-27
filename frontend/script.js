function sendMessage() {
  const input = document.getElementById("user-input");
  const message = input.value.trim();

  if (message === "") return;

  const chatContainer = document.getElementById("chat-container");

  // User message
  const userMessage = document.createElement("div");
  userMessage.classList.add("message", "user");
  userMessage.innerText = message;
  chatContainer.appendChild(userMessage);

  // Fake agent response (for now)
  const agentMessage = document.createElement("div");
  agentMessage.classList.add("message", "agent");
  agentMessage.innerText = "Processing your question about the EPA document...";
  chatContainer.appendChild(agentMessage);

  input.value = "";
}

document
  .getElementById("user-input")
  .addEventListener("keydown", function (event) {
    if (event.key === "Enter") {
      sendMessage();
    }
  });
