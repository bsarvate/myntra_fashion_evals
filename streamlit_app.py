"""Run with .venv/bin/python -m streamlit run streamlit_app.py."""
from pathlib import Path
import hashlib
import io
import json
import os
import time
import uuid

from dotenv import load_dotenv
from PIL import Image
import streamlit as st

from fashion_assistant.agent import ShoppingAgent, nebius_model
from fashion_assistant.catalog import fingerprint
from fashion_assistant.evaluation import save_run
from fashion_assistant.retrieval import BM25, HybridSearch, LocalVectors
from fashion_assistant.validation import Settings

ROOT = Path(__file__).resolve().parent
SUBSET = ROOT / 'data/processed/myntra1000'
ARTIFACTS = ROOT / 'artifacts/myntra1000'
load_dotenv(ROOT / '.env', override=False)
st.set_page_config(page_title='Thread · Fashion assistant', page_icon='🧵', layout='wide')
st.markdown('''<style>
.stApp {background:#faf8f4;} h1,h2,h3 {color:#273b33;}
[data-testid="stSidebar"] {background:#eeeae1;}
[data-testid="stMetric"] {background:white;border:1px solid #e2dfd6;border-radius:12px;padding:16px;}
</style>''', unsafe_allow_html=True)
st.title('Thread')
st.caption('Find your next outfit · 1,000 Myntra products · Text and image search')
st.caption('App revision 5 · inspected alternatives validate automatically')
if not (SUBSET / 'catalog.json').exists():
    st.error('Prepared catalog missing. Add the myntra1000 subset under data/processed.'); st.stop()

@st.cache_data
def catalog():
    return json.loads((SUBSET/'catalog.json').read_text()), json.loads((SUBSET/'inventory.json').read_text())

products, inventory = catalog()
lookup = {p['product_id']: p for p in products}

@st.cache_resource
def keyword_search(revision=3):
    return BM25(products)


def photo(product):
    path = (SUBSET / product.get('image_path', '')).resolve()
    return path if path.is_relative_to(SUBSET.resolve()) and path.is_file() else None


def cards(items):
    for start in range(0, len(items), 3):
        for column, item in zip(st.columns(3), items[start:start+3]):
            p = lookup[item['product_id']]
            with column, st.container(border=True):
                label = {'top': 'Top', 'bottom': 'Bottom', 'set': 'Clothing set'}.get(p['slot'], p['category'])
                st.markdown(f"**{label} · {p['color']} {p['category']}**")
                path = photo(p)
                if path: st.image(str(path), width=220)
                st.caption('Product photo: other garments worn by the model are not included.')
                st.write(p['name'])
                st.caption(f"{p['color']} · {p['fabric']} · {p['category']}")
                st.write(f"{p['price']} catalog units" + (f" · Size {item['size']}" if 'size' in item else ''))


with st.sidebar:
    st.subheader('Your shopping settings')
    st.caption('Historical prices and simulated stock. Size labels do not establish real-world fit.')
    model = st.text_input('Nebius model ID', value='moonshotai/Kimi-K3', key='model')
    request_timeout = st.selectbox('Timeout per model request (seconds)', [60, 120], help='A longer timeout allows slower responses, but can increase waiting time.')
    budget = st.number_input('Outfit budget (catalog units)', min_value=1, value=2500, step=100)
    outfit = st.selectbox('Outfit', ['Top + bottom', 'Set'])
    size = st.selectbox('Top / set size', ['S','M','L'], index=1)
    bottom_size = st.selectbox('Bottom size', ['S','M','L'], index=1, disabled=outfit=='Set')
    st.caption('Optional confirmed requirements')
    color = st.selectbox('Top / set color', ['Any']+sorted({p['color'] for p in products}))
    fabric = st.selectbox('Top / set fabric', ['Any']+sorted({p['fabric'] for p in products}))
    st.caption('Each message can make up to six paid Nebius calls. API keys load from .env.')
    start = st.button('Start new conversation', type='primary', use_container_width=True)

slot = 'set' if outfit=='Set' else 'top'
sizes = {'set':size} if outfit=='Set' else {'top':size,'bottom':bottom_size}
filters = {slot: {k:v for k,v in [('color',color),('fabric',fabric)] if v!='Any'}}
signature = json.dumps([5,model,request_timeout,budget,sizes,filters],sort_keys=True)
if start:
    try:
        shopper = ShoppingAgent(products,inventory,Settings(str(budget),sizes,filters=filters),
                                HybridSearch(keyword_search()),nebius_model(model,timeout=request_timeout),compact_context=True)
        st.session_state.shopper = shopper
        st.session_state.signature = signature
        st.session_state.messages = []
        st.sidebar.success('Conversation ready')
    except Exception as exc:
        st.sidebar.error(str(exc) if isinstance(exc,ValueError) else f'Unable to start: {type(exc).__name__}')

chat, browse, visual, evaluations = st.tabs(['Outfit assistant','Explore catalog','Image search','Evaluation reports'])
with chat:
    st.subheader('An outfit, built around you')
    st.write('Ask for an outfit, then refine it—keep the top, swap the bottom, or explore another color.')
    active = 'shopper' in st.session_state and st.session_state.get('signature')==signature
    if not active: st.info('Confirm your settings with “Start new conversation” to send a message.')
    for i, entry in enumerate(st.session_state.get('messages',[])):
        with st.chat_message('user'): st.write(entry['request'])
        with st.chat_message('assistant'):
            result = entry['result']
            if result['status']=='validated':
                st.success(result['reply'])
                cards(result['proposal']['items'])
                st.write('Subtotal:',result['proposal']['subtotal'],'catalog units')
                st.write(result['proposal'].get('style_suggestion',''))
            elif result['status']=='api_error':
                kind = next((t.get('type') for t in result['trace'] if t.get('event')=='api_error'),'API error')
                st.error(f'{kind}: the request could not finish. No outfit was validated.')
            elif result['status']=='call_limit': st.warning('The agent reached its call limit before validating an outfit.')
            else: st.write(result['reply'])
            st.caption(f"{result['calls']} model calls · {entry['seconds']:.1f} seconds")
            with st.expander('Inspect request trace'):
                st.json(result)
                st.download_button('Download trace',json.dumps(result,indent=2),f'trace_{i}.json',key=f'trace_{i}')
    prompt = st.chat_input('Try: a black cotton top with blue jeans',disabled=not active)
    if prompt:
        started=time.perf_counter()
        with st.status(f'Waiting for {model} · up to six model calls',expanded=True) as status:
            st.write(f'Searching, inspecting products and checking your outfit. Each provider request has a {request_timeout}-second timeout.')
            try:
                st.session_state.shopper.on_progress = lambda message: st.write(message)
                try:
                    result=st.session_state.shopper.chat(prompt)
                finally:
                    st.session_state.shopper.on_progress = None
                entry={'request':prompt,'result':result,'seconds':time.perf_counter()-started}
                st.session_state.messages.append(entry)
                try:
                    save_run(result,ARTIFACTS/'streamlit/conversations',{'model':model,'retrieval':'bm25','compact_context':True})
                except OSError:
                    st.warning('Response retained in this session, but disk saving failed.')
                status.update(label='Request finished',state='complete')
            except Exception as exc:
                status.update(label=f'Request failed: {type(exc).__name__}',state='error')
                st.error('Start a new conversation before retrying.')
                st.session_state.pop('signature',None)
        st.rerun()

with browse:
    st.subheader('Explore the catalog')
    query=st.text_input('Search products',placeholder='Blue jeans or peach polyester top')
    category=st.selectbox('Category',['All']+sorted({p['category'] for p in products}))
    selected={} if category=='All' else {'category':category}
    if query.strip(): rows=keyword_search().search(query,selected,k=24)
    else: rows=[{'product_id':p['product_id']} for p in products if category=='All' or p['category']==category][:24]
    st.caption(f'{len(rows)} products shown · catalog browsing makes no LLM calls')
    if not rows: st.info('No matches. Try another query or category.')
    cards(rows)

with visual:
    st.subheader('Find visually similar pieces')
    st.write('Upload a clothing photograph to search with CLIP. Results show visual similarity; fabric and fit require catalog checks.')
    uploaded=st.file_uploader('Reference photograph',type=['jpg','jpeg','png','webp'])
    if uploaded: st.image(uploaded,width=220)
    st.caption('First use builds a local image index for 1,000 products. It can take several minutes on this Mac. No paid LLM calls are made.')
    if st.button('Find similar products',disabled=uploaded is None):
        try:
            if uploaded.size>10*1024*1024: raise ValueError('Choose an image smaller than 10 MB.')
            content=uploaded.getvalue()
            with Image.open(io.BytesIO(content)) as im: im.verify()
            with st.status('Preparing image search',expanded=True) as status:
                from fashion_assistant.embeddings import CLIPEncoder, CLIP_MODEL_REVISION
                import numpy as np
                if 'clip' not in st.session_state: st.session_state.clip=CLIPEncoder()
                clip=st.session_state.clip
                key=fingerprint({'catalog':fingerprint(products),'revision':CLIP_MODEL_REVISION})
                cache=ARTIFACTS/'streamlit/image_index'/f'{key}.npz'
                ids=[p['product_id'] for p in products]
                if cache.exists():
                    with np.load(cache,allow_pickle=False) as data: vectors=data['vectors']
                else:
                    vectors=[];bar=st.progress(0.0)
                    for offset in range(0,len(products),16):
                        vectors.extend(clip.encode_images([photo(p) for p in products[offset:offset+16]]))
                        bar.progress(min((offset+16)/len(products),1.0),text=f'Indexed {min(offset+16,len(products))}/{len(products)} images')
                    cache.parent.mkdir(parents=True,exist_ok=True)
                    temp=cache.with_name(uuid.uuid4().hex+'.npz');np.savez_compressed(temp,vectors=vectors);temp.replace(cache)
                target=ARTIFACTS/'streamlit/uploads'/(hashlib.sha256(content).hexdigest()+'.image')
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
                store=LocalVectors(products,ids,vectors)
                matches=store.search_vector(clip.encode_images([target])[0],k=9)
                st.session_state.visual_results=matches
                status.update(label='Similar products ready',state='complete')
        except Exception as exc:
            st.error(f'Image search could not finish: {type(exc).__name__}. Check search dependencies and image validity.')
    if 'visual_results' in st.session_state: cards(st.session_state.visual_results)

with evaluations:
    st.subheader('Evaluation evidence')
    st.caption('Saved results only. Opening this page never starts a benchmark or model call.')
    reports=sorted((ARTIFACTS/'nebius').glob('*/report.json'),reverse=True)
    if not reports: st.info('Run a notebook trial to create an evaluation report.')
    else:
        chosen=st.selectbox('Saved run',reports,format_func=lambda p:p.parent.name)
        try:
            report=json.loads(chosen.read_text())
            cols=st.columns(3)
            for col,key,label in zip(cols,['turn_success_rate','dialogue_success_rate','api_error_turns'],['Turn success','Dialogue success','API error turns']):
                value=report.get(key)
                if key=='api_error_turns' and value is None: value=sum(r.get('result',{}).get('status')=='api_error' for r in report.get('rows',[]))
                col.metric(label,'Not recorded' if value is None else f'{value:.0%}' if 'rate' in key else str(value))
            st.write('Run status:',report.get('run_status','Legacy report'))
            st.caption('Small trials and partial runs do not establish overall reliability. Inventory is simulated; relevance labels require human review.')
            st.dataframe([{'case':r.get('case_id'),'turn':r.get('turn',0)+1,'status':r.get('result',{}).get('status',r.get('error')),'success':r.get('success'),'seconds':round(r.get('latency_ms',0)/1000,1)} for r in report.get('rows',[])],hide_index=True)
            st.download_button('Download evaluation JSON',json.dumps(report,indent=2),'evaluation.json')
        except (OSError,ValueError): st.info('Report is being written or is unreadable. Refresh to retry.')
