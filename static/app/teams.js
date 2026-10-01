// Teams page: team cards, the team editor, the character icon library, and the combo's Team dropdown.

function ensureWwAbilityShape(obj) {
    const out = { "1": {}, "2": {}, "3": {} };
    if (!obj || typeof obj !== 'object') return out;
    ['1', '2', '3'].forEach(c => {
        if (obj[c] && typeof obj[c] === 'object') {
            ['e', 'q', 'r'].forEach(a => {
                const url = (obj[c][a] || '').toString().trim();
                if (url) out[c][a] = url;
            });
        }
    });
    return out;
}

function ensureWwSlotShape(obj) {
    const out = { "1": "", "2": "", "3": "" };
    if (!obj || typeof obj !== 'object') return out;
    ['1', '2', '3'].forEach(k => {
        const url = (obj[k] || '').toString().trim();
        if (url) out[k] = url;
    });
    return out;
}

/** Re-render the team and character panels after the editor payload changes. */
function renderWwPanels() {
    renderWwTeamEditor();
    renderWwCharacterEditor();
    renderWwDashPreview();
}

// ----- WW helper: image preview -----
function wwSetPreview(el, val) {
    const v = (val || '').toString().trim();
    if (!v) { el.innerHTML = ''; el.style.display = 'none'; return; }
    el.style.display = 'flex';
    el.style.alignItems = 'center';
    el.style.justifyContent = 'center';
    if (/^https?:\/\//i.test(v)) {
        el.innerHTML = `<img class="key-step-image" src="${escapeHtml(v)}" alt="" loading="lazy" referrerpolicy="no-referrer" style="width:32px;height:32px;object-fit:contain;" />`;
    } else {
        el.innerHTML = `<span class="key-step-emoji">${escapeHtml(v)}</span>`;
    }
}

function wwSetSlotPortrait(el, char) {
    el.replaceChildren();
    const value = (char?.swap_image || '').toString().trim();
    el.classList.toggle('ww-slot-portrait-empty', !value);
    el.title = char?.name ? `${char.name} portrait` : 'Empty character slot';

    if (!value) return;
    if (/^https?:\/\//i.test(value)) {
        const img = document.createElement('img');
        img.src = value;
        img.alt = char?.name ? `${char.name} portrait` : 'Character portrait';
        img.loading = 'lazy';
        img.referrerPolicy = 'no-referrer';
        img.draggable = false;
        el.appendChild(img);
    } else {
        const emoji = document.createElement('span');
        emoji.textContent = value;
        el.appendChild(emoji);
    }
}

/** Team display names that still reference this character (by name_key, case-insensitive). */
function wwTeamNamesReferencingCharacter(nameKey) {
    const key = (nameKey || '').toString().trim().toLowerCase();
    if (!key) return [];
    const names = [];
    (appState.wwTeams || []).forEach(t => {
        if (!t || typeof t !== 'object') return;
        const slots = [t.slot1, t.slot2, t.slot3].map(s => (s || '').toString().trim().toLowerCase());
        if (slots.includes(key)) names.push((t.name || t.id || '').toString() || 'Team');
    });
    return names;
}

// ----- WW Dash preview -----
function renderWwDashPreview() {
    const input = getEl('wwDashImageInput');
    const preview = getEl('wwDashPreview');
    if (input && preview) {
        input.value = (appState.wwDashImage || '').toString();
        wwSetPreview(preview, input.value);
    }
}

// ----- Teams -----
// The Combos page's Team dropdown picks the team a combo uses (appState.wwTeamId / wwTeamSlots).
// The Teams page edits teams on their own (appState.teamEdit), so browsing teams never changes the combo.
function blankTeamEdit() {
    return { id: '', name: '', slots: ['', '', ''], syncedSnap: '', pendingName: '' };
}

function wwCharByKey(key) {
    const k = (key || '').toString().trim().toLowerCase();
    return k ? appState.wwCharacters[k] || null : null;
}

/** Round portrait: the character's swap image when it is a URL, else their initials. */
function makeAvatar(charKey, extraClass) {
    const el = document.createElement('span');
    el.className = 'avatar' + (extraClass ? ' ' + extraClass : '');
    const ch = wwCharByKey(charKey);
    const name = ch?.name || (charKey || '').toString();
    el.title = name || 'Empty slot';
    const img = (ch?.swap_image || '').toString().trim();
    if (/^https?:\/\//i.test(img)) {
        const pic = document.createElement('img');
        pic.src = img;
        pic.alt = '';
        pic.referrerPolicy = 'no-referrer';
        pic.onerror = () => { pic.remove(); el.textContent = name.slice(0, 2).toUpperCase(); };
        el.appendChild(pic);
    } else if (name) {
        el.textContent = name.slice(0, 2).toUpperCase();
    } else {
        el.classList.add('avatar-empty');
    }
    return el;
}

function renderWwTeamEditor() {
    renderTeamSelect();
    syncTeamEditFromServer();
    renderTeamCards();
    renderTeamSlotsEditor();
}

function renderTeamSelect() {
    const teamSelect = getEl('wwTeamSelect');
    if (!teamSelect) return;
    teamSelect.innerHTML = '<option value="">— No team —</option>';
    appState.wwTeams.forEach(t => {
        const opt = document.createElement('option');
        opt.value = t.id;
        opt.textContent = t.name;
        teamSelect.appendChild(opt);
    });
    teamSelect.value = appState.wwTeamId || '';
}

/** Pick up saved changes for the team being edited; unsaved edits survive unrelated refreshes. */
function syncTeamEditFromServer() {
    const te = appState.teamEdit;
    if (!te.id && te.pendingName) {
        const made = appState.wwTeams.find(t => (t.name || '') === te.pendingName);
        if (made) te.id = made.id;
    }
    if (!te.id) return;
    const t = appState.wwTeams.find(x => x.id === te.id);
    if (!t) { appState.teamEdit = blankTeamEdit(); return; }
    const snap = JSON.stringify([t.name, t.slot1, t.slot2, t.slot3]);
    if (snap === te.syncedSnap) return;
    Object.assign(te, { name: t.name || '', slots: [t.slot1 || '', t.slot2 || '', t.slot3 || ''], syncedSnap: snap, pendingName: '' });
}

function editTeam(teamId) {
    appState.teamEdit = blankTeamEdit();
    appState.teamEdit.id = teamId || '';
    syncTeamEditFromServer();
    renderTeamCards();
    renderTeamSlotsEditor();
}

function renderTeamCards() {
    const wrap = getEl('teamCards');
    if (!wrap) return;
    wrap.replaceChildren();
    const counts = {};
    (appState.overview || []).forEach(r => { if (r.team_id) counts[r.team_id] = (counts[r.team_id] || 0) + 1; });
    if (appState.wwTeams.length === 0) {
        const p = document.createElement('p');
        p.className = 'muted';
        p.textContent = 'No teams yet. Use New team to make one.';
        wrap.appendChild(p);
        return;
    }
    appState.wwTeams.forEach(t => {
        const card = document.createElement('button');
        card.type = 'button';
        card.className = 'card team-card' + (t.id === appState.teamEdit.id ? ' on' : '');
        card.title = `Edit ${t.name}`;
        card.addEventListener('click', () => editTeam(t.id));

        const head = document.createElement('div');
        head.className = 'team-card-head';
        const h = document.createElement('span');
        h.className = 'team-card-name';
        h.textContent = t.name;
        const n = counts[t.id] || 0;
        const c = document.createElement('span');
        c.className = 'muted small';
        c.textContent = n === 1 ? '1 combo' : `${n} combos`;
        head.append(h, c);
        card.appendChild(head);

        [t.slot1, t.slot2, t.slot3].forEach((key, i) => {
            const row = document.createElement('div');
            row.className = 'team-slot';
            const cap = document.createElement('span');
            cap.className = 'keycap';
            cap.textContent = String(i + 1);
            const name = document.createElement('span');
            name.textContent = wwCharByKey(key)?.name || (key ? key : 'Empty');
            if (!key) name.className = 'muted';
            row.append(cap, makeAvatar(key), name);
            card.appendChild(row);
        });
        wrap.appendChild(card);
    });
}

function renderTeamSlotsEditor() {
    const te = appState.teamEdit;
    const title = getEl('teamEditorTitle');
    if (title) title.textContent = te.id ? `Edit ${te.name || 'team'}` : 'New team';
    const nameEl = getEl('wwTeamName');
    if (nameEl && document.activeElement !== nameEl) nameEl.value = te.name || '';
    const delBtn = getEl('deleteTeamBtn');
    if (delBtn) delBtn.disabled = !te.id;

    const slotsContainer = getEl('wwTeamSlots');
    if (!slotsContainer) return;
    slotsContainer.innerHTML = '';

    const charOptions = () => {
        const empty = '<option value="">— (empty) —</option>';
        const chars = Object.values(appState.wwCharacters)
            .filter(c => c && c.name)
            .sort((a, b) => (a.name || '').localeCompare(b.name || ''));
        return empty + chars.map(c => `<option value="${escapeHtml(c.name_key || c.name.toLowerCase())}">${escapeHtml(c.name)}</option>`).join('');
    };

    let dragSrcIdx = null;

    te.slots.forEach((charKey, idx) => {
        const row = document.createElement('div');
        row.className = 'ww-slot-row';
        row.draggable = true;
        row.dataset.idx = idx;

        const handle = document.createElement('span');
        handle.className = 'ww-slot-handle';
        handle.textContent = '⠿';
        handle.title = 'Drag to reorder';

        const label = document.createElement('span');
        label.className = 'keycap';
        label.textContent = String(idx + 1);
        label.title = `Slot ${idx + 1}`;

        const portrait = document.createElement('div');
        portrait.className = 'ww-slot-portrait';
        wwSetSlotPortrait(portrait, charKey ? appState.wwCharacters[charKey] : null);

        const sel = document.createElement('select');
        sel.className = 'ww-slot-char-select input';
        sel.setAttribute('aria-label', `Slot ${idx + 1} character`);
        sel.innerHTML = charOptions();
        sel.value = charKey || '';

        sel.addEventListener('change', () => {
            te.slots[idx] = sel.value;
            wwSetSlotPortrait(portrait, sel.value ? appState.wwCharacters[sel.value] : null);
        });

        row.appendChild(handle);
        row.appendChild(label);
        row.appendChild(portrait);
        row.appendChild(sel);
        slotsContainer.appendChild(row);

        row.addEventListener('dragstart', e => {
            dragSrcIdx = idx;
            e.dataTransfer.effectAllowed = 'move';
            setTimeout(() => row.classList.add('ww-drag-active'), 0);
        });
        row.addEventListener('dragend', () => {
            dragSrcIdx = null;
            row.classList.remove('ww-drag-active');
            slotsContainer.querySelectorAll('.ww-slot-row').forEach(r => r.classList.remove('ww-drag-over'));
        });
        row.addEventListener('dragover', e => {
            if (dragSrcIdx !== null && dragSrcIdx !== idx) {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'move';
                slotsContainer.querySelectorAll('.ww-slot-row').forEach(r => r.classList.remove('ww-drag-over'));
                row.classList.add('ww-drag-over');
            }
        });
        row.addEventListener('dragleave', e => {
            if (!row.contains(e.relatedTarget)) row.classList.remove('ww-drag-over');
        });
        row.addEventListener('drop', e => {
            e.preventDefault();
            const src = dragSrcIdx;
            if (src !== null && src !== idx) {
                const tmp = te.slots[src];
                te.slots[src] = te.slots[idx];
                te.slots[idx] = tmp;
                renderTeamSlotsEditor();
            }
        });
    });
}

// ----- WW Character editor -----
function _buildCharRow(labelText, inputValue, onInput) {
    const row = document.createElement('div');
    row.className = 'ww-ability-row';
    const label = document.createElement('span');
    label.className = 'ww-ability-label';
    label.textContent = labelText;
    const input = document.createElement('input');
    input.type = 'text';
    input.placeholder = 'https://... or emoji';
    input.value = inputValue || '';
    const preview = document.createElement('div');
    preview.className = 'ww-ability-preview';
    wwSetPreview(preview, input.value);
    input.addEventListener('input', () => {
        onInput(input.value.trim());
        wwSetPreview(preview, input.value);
    });
    row.appendChild(label);
    row.appendChild(input);
    row.appendChild(preview);
    return row;
}

function renderWwCharacterPicker(chars) {
    const picker = getEl('wwCharPicker');
    if (!picker) return;
    picker.replaceChildren();

    const selectedKey = appState.wwCurrentChar || '';
    const selectedChar = selectedKey ? appState.wwCharacters[selectedKey] : null;
    const trigger = document.createElement('button');
    trigger.type = 'button';
    trigger.className = 'ww-char-picker-trigger';
    trigger.setAttribute('aria-haspopup', 'listbox');
    trigger.setAttribute('aria-expanded', 'false');

    const triggerPortrait = document.createElement('span');
    triggerPortrait.className = 'ww-char-picker-portrait';
    wwSetSlotPortrait(triggerPortrait, selectedChar);
    const triggerLabel = document.createElement('span');
    triggerLabel.className = 'ww-char-picker-label';
    triggerLabel.textContent = selectedChar?.name || '— New Character —';
    const chevron = document.createElement('span');
    chevron.className = 'ww-char-picker-chevron';
    chevron.setAttribute('aria-hidden', 'true');
    trigger.append(triggerPortrait, triggerLabel, chevron);

    const menu = document.createElement('div');
    menu.className = 'ww-char-picker-menu';
    menu.setAttribute('role', 'listbox');
    menu.hidden = true;

    const choose = (key) => {
        appState.wwCurrentChar = key || null;
        renderWwCharacterEditor();
    };
    const addOption = (char) => {
        const key = char ? (char.name_key || char.name.toLowerCase()) : '';
        const option = document.createElement('button');
        option.type = 'button';
        option.className = 'ww-char-picker-option';
        option.setAttribute('role', 'option');
        option.setAttribute('aria-selected', String(key === selectedKey));

        const portrait = document.createElement('span');
        portrait.className = 'ww-char-picker-portrait';
        wwSetSlotPortrait(portrait, char);
        const label = document.createElement('span');
        label.textContent = char?.name || '— New Character —';
        option.append(portrait, label);
        option.addEventListener('click', () => choose(key));
        menu.appendChild(option);
    };

    addOption(null);
    chars.forEach(addOption);
    trigger.addEventListener('click', () => {
        const isOpen = menu.hidden;
        menu.hidden = !isOpen;
        trigger.setAttribute('aria-expanded', String(isOpen));
    });
    trigger.addEventListener('keydown', e => {
        if (e.key === 'Escape') {
            menu.hidden = true;
            trigger.setAttribute('aria-expanded', 'false');
        }
    });

    picker.append(trigger, menu);
}

function renderWwCharacterEditor() {
    const chars = Object.values(appState.wwCharacters)
        .filter(c => c && c.name)
        .sort((a, b) => (a.name || '').localeCompare(b.name || ''));
    renderWwCharacterPicker(chars);

    const container = getEl('wwCharEditor');
    if (!container) return;
    container.innerHTML = '';

    const loaded = appState.wwCurrentChar ? appState.wwCharacters[appState.wwCurrentChar] : null;

    // Name row
    const nameRow = document.createElement('div');
    nameRow.className = 'ww-char-name-row';
    const nameLabel = document.createElement('span');
    nameLabel.className = 'ww-ability-label';
    nameLabel.textContent = 'Name :';
    const nameInput = document.createElement('input');
    nameInput.type = 'text';
    nameInput.id = 'wwCharNameInput';
    nameInput.placeholder = 'Character name';
    nameInput.value = (loaded && loaded.name) ? loaded.name : '';
    nameRow.appendChild(nameLabel);
    nameRow.appendChild(nameInput);
    container.appendChild(nameRow);

    // Character (slot swap) key icon, LMB, Q, E, R rows
    let swapVal = (loaded && loaded.swap_image) || '';
    let lmbVal = (loaded && loaded.lmb_image) || '';
    let abilVals = { q: '', e: '', r: '' };
    if (loaded && loaded.ability_images) {
        abilVals.q = loaded.ability_images.q || '';
        abilVals.e = loaded.ability_images.e || '';
        abilVals.r = loaded.ability_images.r || '';
    }

    container.appendChild(_buildCharRow('Character :', swapVal, v => { swapVal = v; }));
    container.appendChild(_buildCharRow('LMB :', lmbVal, v => { lmbVal = v; }));
    container.appendChild(_buildCharRow('Q :', abilVals.q, v => { abilVals.q = v; }));
    container.appendChild(_buildCharRow('E :', abilVals.e, v => { abilVals.e = v; }));
    container.appendChild(_buildCharRow('R :', abilVals.r, v => { abilVals.r = v; }));

    // Save & Delete buttons
    const btnRow = document.createElement('div');
    btnRow.className = 'ww-char-btn-row';

    const saveBtn = document.createElement('button');
    saveBtn.type = 'button';
    saveBtn.textContent = 'Save';
    saveBtn.addEventListener('click', () => {
        const name = (nameInput.value || '').trim();
        if (!name) { updateStatus('Please enter a character name.', 'fail'); return; }
        const nameKey = name.toLowerCase();
        const isNew = appState.wwCurrentChar === null;
        const overwritingDifferent = !isNew && nameKey !== appState.wwCurrentChar && nameKey in appState.wwCharacters;
        const overwritingExisting = isNew && nameKey in appState.wwCharacters;
        const doSave = () => sendMessage('save_character', {
            name,
            swap_image: swapVal,
            lmb_image: lmbVal,
            ability_images: { q: abilVals.q, e: abilVals.e, r: abilVals.r },
        });
        if (overwritingDifferent || overwritingExisting) {
            if (!confirm(`"${name}" already exists. Overwrite?`)) return;
        }
        doSave();
    });

    const deleteBtn = document.createElement('button');
    deleteBtn.type = 'button';
    deleteBtn.textContent = 'Delete';
    deleteBtn.className = 'danger';
    deleteBtn.disabled = !loaded;
    deleteBtn.addEventListener('click', () => {
        if (!appState.wwCurrentChar) return;
        const charName = (loaded && loaded.name) || appState.wwCurrentChar;
        const blockingTeams = wwTeamNamesReferencingCharacter(appState.wwCurrentChar);
        if (blockingTeams.length) {
            alert(`Remove from these teams first: ${blockingTeams.join(', ')}`);
            return;
        }
        if (!confirm(`Delete character "${charName}"?`)) return;
        sendMessage('delete_character', { name: charName });
    });

    btnRow.appendChild(saveBtn);
    btnRow.appendChild(deleteBtn);
    container.appendChild(btnRow);
}

// WW: one-click character sync from wuthering.gg
const wwSyncCharNameEl = getEl('wwSyncCharName');
function syncCharacterFromWeb() {
    const name = (wwSyncCharNameEl?.value || '').toString().trim();
    if (!name) { updateStatus('Type a character name first.', 'fail'); return; }
    sendMessage('sync_character', { name });
    wwSyncCharNameEl.value = '';
}
getEl('wwSyncCharBtn')?.addEventListener('click', syncCharacterFromWeb);
wwSyncCharNameEl?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); syncCharacterFromWeb(); }
});
getEl('wwSyncAllBtn')?.addEventListener('click', () => sendMessage('sync_all_characters'));

// WW: team select dropdown
document.addEventListener('change', e => {
    if (e.target && e.target.id === 'wwTeamSelect') {
        appState.wwTeamId = (e.target.value || '').toString();
        sendMessage('select_team', { team_id: appState.wwTeamId, target_game: appState.targetGame });
    }
});

// Teams page: save / new / delete act on the team being edited
getEl('wwTeamName')?.addEventListener('input', (e) => { appState.teamEdit.name = e.target.value; });

getEl('saveTeamBtn')?.addEventListener('click', () => {
    const te = appState.teamEdit;
    const name = (getEl('wwTeamName')?.value || '').toString().trim();
    if (!name) { showToast('Give the team a name first.'); return; }
    te.name = name;
    if (!te.id) te.pendingName = name;
    appState.pendingToast = `Saved ${name}`;
    sendMessage('save_team', {
        team_id: te.id || '',
        team_name: name,
        slot1: te.slots[0] || '',
        slot2: te.slots[1] || '',
        slot3: te.slots[2] || '',
    });
});

getEl('newTeamBtn')?.addEventListener('click', () => {
    appState.teamEdit = blankTeamEdit();
    renderTeamCards();
    renderTeamSlotsEditor();
    getEl('wwTeamName')?.focus();
});

const deleteTeamBtn = getEl('deleteTeamBtn');
if (deleteTeamBtn) {
    attachTwoClickConfirm(deleteTeamBtn, {
        confirmText: 'Click again to delete',
        onConfirm: () => {
            const id = appState.teamEdit.id;
            if (!id) return;
            sendMessage('delete_team', { team_id: id });
        }
    });
}

// WW: dash input
document.addEventListener('input', e => {
    if (e.target && e.target.id === 'wwDashImageInput') {
        appState.wwDashImage = e.target.value.trim();
        const preview = getEl('wwDashPreview');
        if (preview) wwSetPreview(preview, e.target.value);
        sendMessage('update_ww_dash', { dash_image: appState.wwDashImage });
        refreshTimelineIfLoaded();
    }
});
