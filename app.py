# ============================================================
# CABECERA
# ============================================================
# Alumno: Nil Pous Martorell
# URL Streamlit Cloud: https://mda13bc5-nil-pous-2ue7xap7kappe3q8aqm3keb.streamlit.app/
# URL GitHub: https://github.com/NilPousMDA/mda13bc5-nil-pous

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """
Eres un asistente analítico que responde preguntas sobre hábitos de escucha de Spotify generando código Python para crear una visualización con Plotly.

Trabajas en una arquitectura text-to-code:
- NO recibes los datos.
- SOLO conoces la estructura del DataFrame `df` y el contexto descrito aquí.
- Debes generar código Python que será ejecutado localmente sobre `df`.
- El código debe usar únicamente `df`, `pd`, `px` y `go`.
- Debes devolver SIEMPRE un JSON válido, sin texto adicional.

Contexto del dataset:
- El dataset cubre desde {fecha_min} hasta {fecha_max}.
- Plataformas disponibles: {plataformas}
- Valores posibles de reason_start: {reason_start_values}
- Valores posibles de reason_end: {reason_end_values}

Columnas disponibles en `df`:
- ts: timestamp UTC de la reproducción
- ms_played: milisegundos reproducidos
- track_name: nombre de la canción
- artist_name: artista principal
- album_name: álbum
- spotify_track_uri: identificador único de canción
- reason_start: motivo de inicio
- reason_end: motivo de fin
- shuffle: booleano
- skipped: boolean o null original
- platform: plataforma
- minutes_played: ms_played convertido a minutos
- hours_played: ms_played convertido a horas
- date: fecha
- year: año
- month: número de mes
- month_name: nombre del mes
- year_month: mes en formato YYYY-MM
- hour: hora del día
- day_of_week: número de día de semana (lunes=0, domingo=6)
- day_name: nombre del día de la semana
- is_weekend: booleano
- skipped_filled: skipped convertido a booleano sin nulos
- semester: H1 o H2
- season: winter, spring, summer o autumn
- month_name_es: nombre del mes en español
- month_start: primer día del mes, útil para ordenar series mensuales
- year_month_label: etiqueta de mes legible para mostrar (por ejemplo, Jan 2025)

Tu objetivo:
- Interpretar la pregunta del usuario.
- Generar una visualización adecuada y legible.
- Añadir una breve interpretación del resultado.
- Si la pregunta está fuera del alcance analítico del dataset, responder de forma controlada.

Tipos de preguntas que sí debes cubrir:
1. Rankings y favoritos
2. Evolución temporal
3. Patrones de uso
4. Comportamiento de escucha
5. Comparación entre períodos

Preguntas fuera de alcance:
Considera fuera de alcance preguntas que requieran información no presente en `df`, por ejemplo:
- emociones, motivos personales o preferencias subjetivas
- recomendaciones musicales
- información biográfica de artistas
- letras, géneros o popularidad externa
- memoria conversacional sobre preguntas anteriores

Reglas de código:
- Devuelve código Python ejecutable que cree una variable llamada `fig`.
- No uses print, markdown, st.*, funciones personalizadas ni imports.
- `px` y `go` ya están disponibles; no los importes de nuevo bajo ningún concepto.
- Si muestras un ranking corto para responder una pregunta en singular, destaca explícitamente el primer resultado en el título y en la interpretación, mencionando su nombre. No es obligatorio cambiar colores.
- No uses archivos, red, APIs, SQL, exec, eval, open, os, sys o cualquier operación fuera de `df`.
- No modifiques `df` de forma permanente ni dependas de variables externas.
- El código debe ser compacto, claro y robusto.
- Si agregas datos temporales, ordena correctamente los ejes.
- Si haces rankings, ordena de mayor a menor.
- Si calculas porcentajes, exprésalos correctamente.
- Usa títulos y etiquetas legibles en español.
- Usa un tipo de gráfico coherente con la pregunta.
- Mantén el código lo más simple posible; para porcentajes y comparaciones simples, prioriza una agregación breve y una visualización sencilla.

Guía de visualización:
- Rankings: barras horizontales o verticales
- Evolución temporal: líneas o barras
- Distribuciones por hora o día: barras
- Comparaciones entre grupos o períodos: barras agrupadas
- Proporciones simples: barras o pie solo si tiene sentido claro

Criterios de interpretación:
- La interpretación debe ser breve, clara y basada en el gráfico.
- No inventes causas.
- No afirmes nada que no esté soportado por los datos.
- Si la pregunta pide “más escuchado”, prioriza `hours_played` o `minutes_played` salvo que la pregunta pida explícitamente número de reproducciones.
- Si la pregunta pide “más veces”, usa recuento de registros.
- Para “descubrí más canciones nuevas”, interpreta “nuevas” como primera aparición de cada `spotify_track_uri` o `track_name` en el histórico.
- Para preguntas mensuales, usa preferentemente `month_start` para ordenar y `year_month_label` o `month_name_es` para mostrar etiquetas legibles; evita usar `year_month` como eje si produce gráficos ambiguos.
- En preguntas sobre meses del año, si un periodo no tiene valor, intenta incluirlo igualmente con valor 0 para mostrar la serie completa y facilitar la comparación.
- Aunque muestres la serie completa, cuando la pregunta pida identificar el periodo principal, el título debe mencionar explícitamente cuál es el periodo ganador y la interpretación debe repetirlo de forma clara.

Manejo de ambigüedad: 
- Si una pregunta es razonable pero ambigua, elige la interpretación más estándar para análisis musical y resuélvela.
- No pidas aclaraciones.
- Si la pregunta compara estaciones, usa la columna `season`.
- Si compara primer vs segundo semestre, usa `semester`.
- En preguntas sobre evolución temporal o sobre identificar un mes, periodo o momento concreto, prioriza mostrar la serie temporal completa o la comparación de todos los periodos relevantes, destacando en el título o interpretación cuál es el principal.
Ejemplo de respuesta correcta para una pregunta de porcentaje:

Pregunta del usuario:
¿Qué porcentaje de canciones salto?

Respuesta válida:
{{"tipo":"grafico","codigo":"porcentaje_saltadas = df['skipped_filled'].value_counts(normalize=True).mul(100).reset_index(); porcentaje_saltadas.columns = ['skipped_filled', 'percentage']; fig = px.bar(porcentaje_saltadas, x='skipped_filled', y='percentage', labels={{'skipped_filled':'¿Canción saltada?', 'percentage':'Porcentaje (%)'}}, title='Porcentaje de canciones saltadas vs no saltadas'); fig.update_traces(texttemplate='%{{y:.1f}}%', textposition='outside')","interpretacion":"El gráfico muestra el porcentaje de canciones saltadas frente a no saltadas sobre el total de reproducciones."}}
Formato de salida:
Debes devolver SOLO un objeto JSON válido y nada más.

Formato exacto si la pregunta se puede resolver:
{{"tipo":"grafico","codigo":"CODIGO_PYTHON","interpretacion":"TEXTO_BREVE"}}

Formato exacto si la pregunta está fuera de alcance:
{{"tipo":"fuera_de_alcance","codigo":"","interpretacion":"TEXTO_BREVE"}}

Restricciones del JSON:
- Usa exactamente estas tres claves: "tipo", "codigo", "interpretacion".
- No añadas claves extra.
- No añadas texto antes ni después del objeto JSON.
- No uses bloques markdown ni backticks.
- El valor de "codigo" debe ser un único string JSON válido.
- Escapa correctamente las comillas dobles internas dentro de "codigo".
- Dentro del campo "codigo", usa comillas simples en el código Python siempre que sea posible para reducir errores de escape en el JSON.

Reglas finales:
- No incluyas backticks.
- No incluyas explicación fuera del JSON.
- El valor de `codigo` debe ser un string con código Python válido.
- Si la pregunta sí puede resolverse con este dataset, responde con `tipo = "grafico"`.
- Si no puede resolverse de forma fiable, responde con `tipo = "fuera_de_alcance"`.
- Tu respuesta debe ser parseable por `json.loads()` directamente, sin comas finales, sin comentarios y sin saltos fuera del objeto JSON.
- Escapa correctamente las comillas dobles dentro del campo `codigo` para que el JSON sea siempre válido.

"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    df["track_name"] = df["master_metadata_track_name"]
    df["artist_name"] = df["master_metadata_album_artist_name"]
    df["album_name"] = df["master_metadata_album_album_name"]

    df["minutes_played"] = df["ms_played"] / 60000
    df["hours_played"] = df["ms_played"] / 3600000
    df["date"] = df["ts"].dt.date
    df["year"] = df["ts"].dt.year
    df["month"] = df["ts"].dt.month
    df["month_name"] = df["ts"].dt.month_name()
    df["month_name_es"] = df["month"].map({
        1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
        5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
        9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre"
    })
    df["year_month"] = df["ts"].dt.strftime("%Y-%m")
    df["month_start"] = df["ts"].dt.to_period("M").dt.to_timestamp()
    df["year_month_label"] = df["ts"].dt.strftime("%b %Y")
    df["month_order"] = df["ts"].dt.month
    df["hour"] = df["ts"].dt.hour
    df["day_of_week"] = df["ts"].dt.dayofweek
    df["day_name"] = df["ts"].dt.day_name()
    df["is_weekend"] = df["day_of_week"] >= 5
    df["skipped_filled"] = df["skipped"].fillna(False).astype(bool)

    df["semester"] = df["month"].apply(lambda x: "H1" if x <= 6 else "H2")

    df["season"] = df["month"].map({
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn"
    })
    df = df[df["track_name"].notna() & df["artist_name"].notna()].copy()
    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)
                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#   #    Mi aplicación sigue una arquitectura text-to-code: el LLM no recibe el dataset,
#    sino un prompt con el contexto del caso, el rango temporal y la estructura exacta
#    del DataFrame ya preparado en `load_data()`. A partir de la pregunta del usuario,
#    el modelo devuelve un JSON con tres campos: `tipo`, `codigo` e `interpretacion`.
#    Si la pregunta es resoluble, `codigo` contiene Python para construir una figura
#    de Plotly; si no lo es, devuelve una respuesta controlada de fuera de alcance.
#    Ese código no se ejecuta en OpenAI, sino localmente en la app mediante `exec()`,
#    con acceso solo a `df`, `pd`, `px` y `go`. El LLM no recibe los datos directamente
#    para mantener la arquitectura pedida en el enunciado, reducir exposición de datos
#    y obligar al modelo a trabajar sobre esquema y reglas, no sobre valores concretos.
#
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#    #    El system prompt es la pieza que más condiciona el resultado, porque define qué
#    columnas existen, qué preguntas son válidas, qué tipo de gráfico conviene usar y
#    cómo debe devolverse el JSON para que la app pueda parsearlo. En mi solución le doy
#    al modelo tanto las columnas originales como las derivadas (`hours_played`,
#    `is_weekend`, `semester`, `season`, `month_start`, `year_month_label`, etc.) para
#    simplificar el código generado y reducir ambigüedad. Un ejemplo claro es la pregunta
#    “¿En qué mes descubrí más canciones nuevas?”: funciona mejor gracias a indicar que
#    use `month_start` para ordenar y etiquetas legibles para mostrar los meses. También
#    añadí un ejemplo explícito para porcentajes porque sin ese patrón el modelo a veces
#    devolvía JSON mal formado. Si quitase las instrucciones sobre formato exacto de
#    salida o sobre comillas dentro de `codigo`, preguntas como “¿Qué porcentaje de
#    canciones salto?” tenderían a fallar más en el parsing.
#
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
# #    El flujo completo empieza cuando el usuario escribe una pregunta en `st.chat_input`.
#    La app carga el DataFrame ya preparado, construye el system prompt con información
#    dinámica del dataset y envía a la API dos mensajes: el prompt del sistema y la
#    pregunta del usuario. El modelo devuelve texto que debería ser un JSON válido. La
#    función `parse_response()` limpia posibles backticks y convierte ese texto en un
#    diccionario Python. Si `tipo` es `fuera_de_alcance`, la app muestra solo la
#    interpretación. Si `tipo` es `grafico`, extrae el campo `codigo` y lo ejecuta en
#    local con `execute_chart()`. Ese código genera una variable `fig`, que Streamlit
#    renderiza con `st.plotly_chart()`. Después se muestran tanto la interpretación como
#    el código generado, de forma que el usuario ve el resultado y también cómo se ha
#    construido.