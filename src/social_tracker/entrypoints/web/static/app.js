// When an in-page request cannot reach the server, say so in the status region.
// Text only: nothing from the page or the network is ever inserted as markup.
document.addEventListener("htmx:sendError", function () {
  var region = document.getElementById("live-region");
  var message = document.createElement("p");
  message.className = "error";
  message.textContent =
    "The server cannot be reached. The list may be out of date; reload the page when it is back.";
  region.replaceChildren(message);
});

// A form that must be sent once (add, edit, delete, the global switch) disables its button as it
// is submitted, so a double click or a double Enter cannot send it twice. The server is safe
// either way; this only spares the second, confusing answer. Without script the form still works.
document.addEventListener("submit", function (event) {
  var form = event.target;
  if (!(form instanceof HTMLFormElement) || !form.hasAttribute("data-once")) {
    return;
  }
  // After the browser has read the form: a disabled submitter is not part of the data.
  window.setTimeout(function () {
    form.querySelectorAll('button[type="submit"]').forEach(function (button) {
      button.disabled = true;
    });
  }, 0);
});

// Coming back through the history restores the page as it was left: give the buttons back.
window.addEventListener("pageshow", function (event) {
  if (event.persisted) {
    document.querySelectorAll("form[data-once] button[type='submit']").forEach(function (button) {
      button.disabled = false;
    });
  }
});
