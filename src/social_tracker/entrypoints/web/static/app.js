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
