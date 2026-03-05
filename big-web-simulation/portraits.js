function generatePortraits(options) {
  const {
    containerSelector = ".gallery-grid",
    count = 100,
    siteLabel = "Site",
    seed,
  } = options || {};

  const container = document.querySelector(containerSelector);
  if (!container) return;

  const effectiveSeed =
    seed || String(siteLabel || "site").replace(/\s+/g, "").toLowerCase();

  const createCard = (user, index) => {
    const gender = user.gender === "male" ? "male" : "female";
    const imgUrl = user.picture && user.picture.large ? user.picture.large : "";
    const name =
      user.name && user.name.first && user.name.last
        ? `${user.name.first} ${user.name.last}`
        : `Person ${index + 1}`;

    const card = document.createElement("article");
    card.className = "portrait-card";

    const img = document.createElement("img");
    img.src = imgUrl;
    img.alt = `Portrait of ${name}`;
    img.loading = "lazy";

    const info = document.createElement("div");
    info.className = "portrait-info";

    const nameEl = document.createElement("div");
    nameEl.className = "portrait-name";
    nameEl.textContent = name;

    const metaEl = document.createElement("div");
    metaEl.className = "portrait-meta";
    metaEl.textContent = `${
      gender === "male" ? "Man" : "Woman"
    } · ${siteLabel} · #${index + 1}`;

    info.appendChild(nameEl);
    info.appendChild(metaEl);
    card.appendChild(img);
    card.appendChild(info);

    return card;
  };

  const url = `https://randomuser.me/api/?results=${count}&seed=${encodeURIComponent(
    effectiveSeed
  )}&inc=gender,name,picture`;

  fetch(url)
    .then((res) => res.json())
    .then((data) => {
      const results = Array.isArray(data.results) ? data.results : [];
      const frag = document.createDocumentFragment();

      results.forEach((user, index) => {
        frag.appendChild(createCard(user, index));
      });

      container.appendChild(frag);
    })
    .catch(() => {
      // If the API fails, leave the gallery empty rather than breaking the page.
    });
}

