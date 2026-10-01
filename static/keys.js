// Live key overlay: lights up keys as the backend reports them (type "key_input"),
// and labels 1/2/3 with the selected team's characters.
const keyEls = new Map(Array.from(document.querySelectorAll('.key')).map((el) => [el.dataset.key, el]));

function setDown(key, down) {
    const el = keyEls.get(key);
    if (el) el.classList.toggle('down', !!down);
}

function releaseAll() {
    keyEls.forEach((el) => el.classList.remove('down'));
}

/** Show character names (and portraits) of the selected team on the swap keys. */
function applyTeam(editor) {
    if (!editor || !editor.ww_team_slots) return;
    const chars = new Map((editor.ww_characters || []).map((c) => [c.name_key, c]));
    ['1', '2', '3'].forEach((slot) => {
        const el = keyEls.get(slot);
        if (!el) return;
        const ch = chars.get(editor.ww_team_slots[`slot${slot}`] || '');
        el.querySelector('.role').textContent = ch ? ch.name : `Slot ${slot}`;
        el.querySelector('.portrait')?.remove();
        const img = (ch && ch.swap_image) || '';
        if (/^https?:\/\//i.test(img)) {
            const pic = document.createElement('img');
            pic.className = 'portrait';
            pic.src = img;
            pic.alt = '';
            pic.referrerPolicy = 'no-referrer';
            pic.onerror = () => pic.remove();
            el.prepend(pic);
        }
    });
}

connectTracker({
    onOpen: () => keyEls.forEach((el) => el.classList.remove('offline')),
    onClose: () => {
        releaseAll();
        keyEls.forEach((el) => el.classList.add('offline'));
    },
    onMessage: (msg) => {
        if (msg.type === 'key_input') setDown(msg.key, msg.down);
        else if (msg.type === 'init') applyTeam(msg.editor);
        else if (msg.type === 'combo_data') applyTeam(msg);
    },
});
