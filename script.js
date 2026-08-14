const API_URL = "https://deepresearch-agent-multi-agent-technical.onrender.com";

const form = document.getElementById("blogForm");
const topicInput = document.getElementById("topicInput");
const submitBtn = document.getElementById("submitBtn");
const outputBody = document.getElementById("outputBody");
const copyBtn = document.getElementById("copyBtn");
const insightsPanel = document.getElementById("insightsPanel");
const modeBadge = document.getElementById("modeBadge");
const queriesList = document.getElementById("queriesList");

let generatedMarkdown = "";
let isGenerating = false;

// =========================================================
// Marked.js Configuration (Makes Links Clickable & Open in New Tab)
// =========================================================
if (window.marked) {
    const renderer = new marked.Renderer();
    renderer.link = ({ href, title, text }) => {
        return `<a href="${href}" title="${title || ''}" target="_blank" rel="noopener noreferrer">${text}</a>`;
    };
    marked.setOptions({ renderer });
}

// =========================================================
// Generate Blog
// =========================================================

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    e.stopPropagation();

    console.count("Generate request");

    // Prevent duplicate requests
    if (isGenerating) {
        console.log("Generation already running. Ignoring duplicate request.");
        return;
    }

    const topic = topicInput.value.trim();

    if (!topic) {
        return;
    }

    isGenerating = true;

    // Keep the topic visible
    topicInput.disabled = true;

    submitBtn.disabled = true;
    submitBtn.innerText = "Running Agent Pipeline...";

    copyBtn.classList.add("hidden");
    insightsPanel.classList.add("hidden");

    outputBody.innerHTML = `
        <div class="placeholder-text loading-state">
            <div class="spinner"></div>
            <p>Agent is routing, researching and writing...</p>
        </div>
    `;

    try {
        console.log("Sending request for:", topic);

        const response = await fetch(`${API_URL}/api/generate`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                topic: topic
            })
        });

        console.log("Backend response status:", response.status);

        if (!response.ok) {
            let errorMessage = "Backend request failed.";
            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || errorMessage;
            } catch (_) {
                // Ignore JSON parsing error
            }
            throw new Error(errorMessage);
        }

        const result = await response.json();

        console.log("Generation completed.");

        generatedMarkdown = result.content || "";

        if (!generatedMarkdown) {
            throw new Error("Backend returned an empty response.");
        }

        // =================================================
        // Display Output (Renders Clickable HTML Links)
        // =================================================
        if (window.marked) {
            outputBody.innerHTML = marked.parse(generatedMarkdown);
        } else {
            outputBody.textContent = generatedMarkdown;
        }

        copyBtn.classList.remove("hidden");

        // =================================================
        // Execution Insights
        // =================================================
        modeBadge.textContent = (result.mode || "closed_book")
            .replaceAll("_", " ")
            .toUpperCase();

        queriesList.innerHTML = "";

        if (result.queries && result.queries.length > 0) {
            result.queries.forEach((query) => {
                const li = document.createElement("li");
                li.textContent = query;
                queriesList.appendChild(li);
            });
        } else {
            queriesList.innerHTML = `
                <li>
                    No web research was required.
                </li>
            `;
        }

        insightsPanel.classList.remove("hidden");

    } catch (error) {
        console.error("Generation error:", error);

        outputBody.innerHTML = `
            <div class="error-state">
                <div class="error-icon">!</div>
                <h3>Unable to generate article</h3>
                <p>
                    ${escapeHtml(
                        error.message || "Could not connect to the backend."
                    )}
                </p>
            </div>
        `;
    } finally {
        isGenerating = false;
        topicInput.disabled = false;
        submitBtn.disabled = false;
        submitBtn.innerText = "Generate Content";
    }
});

// =========================================================
// Copy Markdown
// =========================================================

copyBtn.addEventListener("click", async () => {
    if (!generatedMarkdown) {
        return;
    }

    try {
        await navigator.clipboard.writeText(generatedMarkdown);

        copyBtn.innerText = "Copied!";

        setTimeout(() => {
            copyBtn.innerText = "Copy Raw Markdown";
        }, 2000);

    } catch (error) {
        console.error("Copy failed:", error);
    }
});

// =========================================================
// Escape HTML
// =========================================================

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}