import test from 'node:test';
import assert from 'node:assert/strict';

import {
    applyVoiceCatalogAttribution,
    describeVoiceCatalog,
} from '../../kestrel_sovereign/static/js/voice/provider-attribution.js';

const PROVIDER_NAMES = { openai: 'OpenAI', xai: 'xAI' };

test('xAI without its realtime provider labels the OpenAI Pipeline catalog as fallback', () => {
    const route = {
        path: 'pipeline',
        llm_vendor: 'xai',
        tts_provider: 'openai',
        reason: "Realtime unavailable: no conversation provider declares LLM vendor 'xai'.",
    };

    const result = describeVoiceCatalog(route, 'openai', 'auto', PROVIDER_NAMES);

    assert.equal(result.label, 'OpenAI fallback');
    assert.equal(result.isFallback, true);
    assert.match(result.title, /Realtime unavailable/);
});

test('installed xAI realtime route identifies its own catalog without fallback', () => {
    const route = {
        path: 'realtime',
        conversation_provider: 'xai_realtime',
        conversation_capabilities: { xai_realtime: { vendor: 'xai' } },
    };

    assert.deepEqual(
        describeVoiceCatalog(route, 'xai_realtime', 'auto', PROVIDER_NAMES),
        {
            label: 'xAI Realtime',
            title: 'xAI owns this realtime voice catalog and the full voice turn.',
            isFallback: false,
        },
    );
});

test('OpenAI realtime and explicit OpenAI Pipeline paths are not mislabeled as fallback', () => {
    const realtime = describeVoiceCatalog({
        path: 'realtime',
        conversation_capabilities: { openai_realtime: { vendor: 'openai' } },
    }, 'openai_realtime', 'realtime', PROVIDER_NAMES);
    const pipeline = describeVoiceCatalog(
        { path: 'pipeline' }, 'openai', 'pipeline', PROVIDER_NAMES,
    );

    assert.equal(realtime.label, 'OpenAI Realtime');
    assert.equal(realtime.isFallback, false);
    assert.equal(pipeline.label, 'OpenAI Pipeline');
    assert.equal(pipeline.isFallback, false);
});

test('providerless auto route still exposes that Pipeline is a fallback', () => {
    const result = describeVoiceCatalog(
        { path: 'pipeline', reason: 'No matching realtime provider.' },
        '',
        'auto',
        PROVIDER_NAMES,
    );

    assert.equal(result.label, 'Pipeline fallback');
    assert.equal(result.isFallback, true);
    assert.equal(result.title, 'No matching realtime provider.');
});

test('DOM attribution update renders and styles the fallback state', () => {
    const classes = new Set();
    const element = {
        textContent: '',
        title: '',
        hidden: true,
        classList: {
            toggle(name, enabled) {
                if (enabled) classes.add(name);
                else classes.delete(name);
            },
        },
    };

    applyVoiceCatalogAttribution(
        element,
        { path: 'pipeline', reason: 'xAI realtime provider is unavailable.' },
        'openai',
        'auto',
        PROVIDER_NAMES,
    );

    assert.equal(element.textContent, '· OpenAI fallback');
    assert.equal(element.hidden, false);
    assert.equal(element.title, 'xAI realtime provider is unavailable.');
    assert.equal(classes.has('is-fallback'), true);
});
