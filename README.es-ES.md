# 【ICML 2026】 Furina: Fragmented Uncertainty-Driven Refusal Instability Attack

> [!WARNING]
> **Solo para investigación académica y evaluación autorizada de red-teaming. No apoyamos ni condonamos el mal uso.**

$${\color{red}\text{\textbf{Advertencia: ¡Este repositorio contiene respuestas dañinas del modelo!!!}}}$$

Este repositorio contiene el código para un pipeline de jailbreak / red-teaming de múltiples etapas basado en descomposición, razonamiento, sondeo (probing), recolección de respuestas, síntesis y juzgamiento. Soporta tanto un pipeline solo de texto como un pipeline aumentado con visión.

📄 Paper: [Furina: Fragmented Uncertainty-Driven Refusal Instability Attack](https://arxiv.org/abs/2605.26158)

Advertencia: este repositorio acompaña a un proyecto de investigación de jailbreak / evaluación de seguridad y puede producir salidas dañinas u ofensivas durante la evaluación.

Los dos puntos de entrada principales son:
- [pipeline_runner.py](pipeline_runner.py): pipeline de texto
- [vision_pipeline_runner.py](vision_pipeline_runner.py): pipeline de visión

## Resumen (Abstract)

**Furina** está motivada por la observación de que la alineación de seguridad en los LLMs y MLLMs a menudo no se comporta como un umbral binario limpio. Cerca de los límites de rechazo, pequeñas perturbaciones semánticas o estructurales pueden mover al modelo hacia regiones de inestabilidad donde el rechazo, el cumplimiento parcial y el cumplimiento total se vuelven resultados estocásticos en lugar de deterministas.

Este repositorio implementa la visión del pipeline detrás de esa idea: descomposición fragmentada, sondeo consciente de la etapa, síntesis de respuestas y una rama de visión basada en tipografía o generación de imágenes seguida de análisis visual. El objetivo general no es la optimización de tokens adversarios específicos del modelo, sino la inducción de inestabilidad a través de contextos fragmentados, anclados a escenas y que amplifican la incertidumbre.

## Figura del Método

![Furina overview](Furina_overview.png)

## Descripción General

El pipeline de texto ejecuta estas etapas:
- desglose de tareas (task breakdown)
- razonamiento de sondeo (probe reasoning)
- generación de sondeo (probe generation)
- respuesta al sondeo (probe answering)
- síntesis de red-team (red-team synthesis)
- juzgamiento de política (policy judging)

El pipeline de visión añade una rama visual antes de las etapas de texto:
- generación de prompts de tipografía, o generación de imagen mediante tipografía + Stable Diffusion
- análisis visual de las imágenes generadas
- síntesis consciente de la visión
- la misma etapa de juzgamiento posterior

## Estructura del Repositorio

- `pipeline_runner.py`: ejecutor del pipeline de texto con un solo comando
- `vision_pipeline_runner.py`: ejecutor del pipeline de visión con un solo comando
- `utils/`: módulos del pipeline de texto
- `vision_utils/`: módulos de generación de imágenes, análisis visual y síntesis de visión
- `Results/`: salidas predeterminadas del pipeline de texto
- `Vision_Results/`: salidas predeterminadas del pipeline de visión
- [utils/api_client.py](utils/api_client.py): configuración predeterminada compartida de proveedores y agentes
- [.env.example](.env.example): plantilla de variables de entorno
- [requirements.txt](requirements.txt): dependencias de Python

## Configuración del Entorno

### 1. Crear un entorno de Python

```bash
conda create -n furina python=3.10 -y
conda activate furina
cd Furina_Jailbreak
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

Las dependencias actuales incluyen:
- `openai`
- `python-dotenv`
- `Pillow`
- `torch`
- `torchvision`
- `transformers`
- `tokenizers`
- `diffusers`

Notas:
- `torch` / `diffusers` solo son necesarios cuando se utiliza la ruta de imagen de Stable Diffusion en el pipeline de visión.
- El modo de visión predeterminado `typo` no requiere un modelo de difusión local.

### 3. Crear `.env`

```bash
cp .env.example .env
```

Luego, complete las credenciales que realmente utilice.

## Configuración del Proveedor

La lógica compartida del proveedor y el enrutamiento predeterminado del agente residen en [utils/api_client.py](utils/api_client.py).

El proyecto soporta actualmente estos proveedores compatibles con OpenAI:
- OpenAI
- DeepSeek
- Gemini
- Grok
- Claude
- Proxy / relay

Las variables de entorno se definen en [.env.example](.env.example):

```bash
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1

DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com

GEMINI_API_KEY=
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/

GROK_API_KEY=
GROK_BASE_URL=https://api.groq.com/openai/v1

CLAUDE_API_KEY=
CLAUDE_BASE_URL=https://api.anthropic.com

PROXY_API_KEY=
PROXY_BASE_URL=
```

### Valores Predeterminados del Agente

Los valores predeterminados actuales del pipeline de texto en `AGENT_DEFAULTS` son:
- `task_plan` -> `openai / gpt-4o-mini`
- `probe_reasoning` -> `openai / o4-mini`
- `probe_optimizer` -> `openai / gpt-4o-mini`
- `probe_generator` -> `openai / gpt-4o-mini`
- `probe_responder` -> `openai / gpt-4o-mini`
- `redteam_synthesizer` -> `deepseek / deepseek-v4-pro`
- `redteam_judge` -> `openai / gpt-4o`

Reglas de enrutamiento importantes:
- Si un agente tiene un proveedor explícito como `openai`, `deepseek` o `proxy`, ese proveedor se mantiene.
- La inferencia automática del proveedor a partir de los prefijos del nombre del modelo solo ocurre cuando el proveedor está configurado como `auto`.
- En el pipeline de visión, el enrutamiento predeterminado del analizador de visión sigue la configuración del agente `probe_responder`.
- En el pipeline de visión, el enrutamiento predeterminado del sintetizador de visión sigue la configuración del agente `redteam_synthesizer`.

## Formato de Entrada

Prepare un archivo de texto plano con una tarea por línea.

Ejemplo `tasks.txt`:

```txt
task one
task two
task three
```

Las líneas en blanco son ignoradas por la etapa de preprocesamiento de visión.

## Pipeline de Texto

### Ejecución Básica

```bash
python pipeline_runner.py -i tasks.txt
```

### Qué hace

El pipeline de texto ejecuta:
1. `task_plan.batch_analyze_from_txt(...)`
2. `probe_reasoning_agent.process_batch(...)`
3. `phase_probe_generator.process_batch(...)`
4. reparación opcional para salidas de sondeo `SKIP` / faltantes
5. `probe_responder.process_batch(...)`
6. `redteam_synthesizer.process_batch(...)`
7. `redteam_judge.process_batch(...)`

### Directorios de Salida Predeterminados

El pipeline de texto escribe en:
- `Results/mission_breakdown_reports/`
- `Results/probe_reasoning_results/`
- `Results/phase_probe_results/`
- `Results/probe_responses/`
- `Results/redteam_synthesized/`
- `Results/redteam_judged/`

### Sobrescritura Común de Modelos

```bash
python pipeline_runner.py \
  -i tasks.txt \
  --task-plan-model gpt-4o-mini \
  --reasoning-model o4-mini \
  --optimizer-model gpt-4o-mini \
  --generator-model gpt-4o-mini \
  --responder-model gpt-4o-mini \
  --synthesizer-model deepseek-v4-pro \
  --judge-model gpt-4o
```

### Deshabilitar Paso de Reparación

Por defecto, el pipeline de texto intenta reparar las salidas de los sondeos cuando una tarea recibe un `SKIP` o no tiene preguntas de sondeo generadas.

Para deshabilitar esto:

```bash
python pipeline_runner.py -i tasks.txt --no-repair-skips
```

## Pipeline de Visión

### Ejecución Básica

El modo predeterminado es `typo`:

```bash
python vision_pipeline_runner.py -i tasks.txt
```

Este modo:
- genera prompts visuales de estilo tipográfico a partir de cada línea de entrada
- los renderiza en imágenes
- analiza esas imágenes con un modelo de visión
- continúa hacia el pipeline de razonamiento y juzgamiento posterior estilo texto

### Modos de Visión

Hay dos modos:
- `typo`: texto tipográfico -> imagen renderizada
- `sd`: texto tipográfico -> imagen de Stable Diffusion

Ejecute el modo Stable Diffusion con:

```bash
python vision_pipeline_runner.py -i tasks.txt --vision-image-mode sd
```

En el modo `sd`, el pipeline sigue generando primero el texto tipográfico, y luego utiliza ese texto como fuente del prompt para Stable Diffusion.

### Etapas del Pipeline de Visión

El ejecutor de visión hace lo siguiente:
1. generación de tipografía, o generación de tipografía + Stable Diffusion
2. `vision_analyzer.batch_analyze_vision(...)`
3. `task_plan.batch_analyze_from_txt(...)`
4. `probe_reasoning_agent.process_batch(...)`
5. `phase_probe_generator.process_batch(...)`
6. reparación opcional para salidas de sondeo `SKIP` / faltantes
7. `probe_responder.process_batch(...)`
8. `vision_redteam_synthesizer.process_batch(...)`
9. `redteam_judge.process_batch(...)`

### Directorios de Salida de Visión

El pipeline de visión escribe en:
- `Vision_Results/typo_images/`
- `Vision_Results/typography_texts/`
- `Vision_Results/sd_images/`
- `Vision_Results/vision_inference/`
- `Vision_Results/mission_breakdown_reports/`
- `Vision_Results/probe_reasoning_results/`
- `Vision_Results/phase_probe_results/`
- `Vision_Results/probe_responses/`
- `Vision_Results/vision_redteam_synthesized/`
- `Vision_Results/redteam_judged/`

### Configuración de Tipografía

Puede controlar la API de generación de tipografía directamente desde el ejecutor:

```bash
python vision_pipeline_runner.py \
  -i tasks.txt \
  --typography-model deepseek-chat \
  --typography-api-env-key DEEPSEEK_API_KEY \
  --typography-base-url https://api.deepseek.com
```

Si utiliza un proxy:

```bash
python vision_pipeline_runner.py \
  -i tasks.txt \
  --typography-model gpt-4o-mini \
  --typography-api-env-key PROXY_API_KEY \
  --typography-base-url https://your-proxy.example.com/v1
```

### Configuración del Analizador de Visión

Por defecto, el analizador de visión sigue la misma ruta de destino que `--responder-model`. Esto es intencional: el mismo modelo de destino se utiliza tanto para la interpretación visual como para la respuesta al sondeo posterior.

Por lo tanto, en el caso común, cambiar `--responder-model` es suficiente:

```bash
python vision_pipeline_runner.py \
  -i tasks.txt \
  --responder-model gpt-4o-mini
```

Si `probe_responder` está enrutado a través de un proxy en `utils/api_client.py`, el analizador de visión seguirá esa misma ruta de proveedor por defecto.

Aun así, puede sobrescribir el destino del análisis visual independientemente cuando sea necesario:

```bash
python vision_pipeline_runner.py \
  -i tasks.txt \
  --vision-model gpt-4o-mini \
  --vision-api-env-key PROXY_API_KEY \
  --vision-base-url https://your-proxy.example.com/v1
```

### Configuración del Sintetizador de Visión

Por defecto, el sintetizador de visión sigue la configuración de `redteam_synthesizer` en `utils/api_client.py`.

Puede sobrescribirlo desde la CLI:

```bash
python vision_pipeline_runner.py \
  -i tasks.txt \
  --vision-synthesizer-model deepseek-v4-pro \
  --vision-synthesizer-api-env-key DEEPSEEK_API_KEY \
  --vision-synthesizer-base-url https://api.deepseek.com
```

### Configuración de Stable Diffusion

Si utiliza `--vision-image-mode sd`, es probable que necesite sobrescribir la ruta del modelo local:

```bash
python vision_pipeline_runner.py \
  -i tasks.txt \
  --vision-image-mode sd \
  --sd-model-path /path/to/stabilityai/stable-diffusion-xl-base-1.0
```

Parámetros adicionales de SD:
- `--sd-guidance-scale`
- `--sd-num-inference-steps`
- `--sd-width`
- `--sd-height`

### Registros (Logs) del Pipeline de Visión

`vision_pipeline_runner.py` ahora imprime bloques de progreso por etapa para:
- generación de tipografía
- análisis de visión
- etapas de texto posteriores que ya tienen sus propios registros

Por lo tanto, debería ver mensajes explícitos de `Completed` antes de que el pipeline entre en las etapas posteriores de razonamiento y juzgamiento.

## Inicio Rápido

### Inicio Rápido del Pipeline de Texto

```bash
conda create -n furina python=3.10 -y
conda activate furina
pip install -r requirements.txt
cp .env.example .env
python pipeline_runner.py -i tasks.txt
```

### Inicio Rápido del Pipeline de Visión

```bash
conda create -n furina python=3.10 -y
conda activate furina
pip install -r requirements.txt
cp .env.example .env
python vision_pipeline_runner.py -i tasks.txt
```

## Problemas Comunes

### 1. `Missing environment variable: OPENAI_API_KEY`

Esto significa que la etapa seleccionada está intentando usar el proveedor de OpenAI, pero su archivo `.env` no contiene `OPENAI_API_KEY`.

Opciones de solución:
- completar `OPENAI_API_KEY` en `.env`
- cambiar esa etapa a `PROXY_API_KEY` y `PROXY_BASE_URL`
- cambiar el proveedor/modelo predeterminado en `utils/api_client.py`

### 2. `bitsandbytes was compiled without GPU support`

Esta advertencia proviene del stack de dependencias de Stable Diffusion. No es fatal por sí misma.

Si está utilizando el modo `typo` predeterminado, el stack de SD no es necesario. Si está utilizando el modo `sd`, verifique su instalación local de PyTorch / CUDA / bitsandbytes.

### 3. Errores de ruta del modelo de Stable Diffusion

Si utiliza `--vision-image-mode sd`, la ruta predeterminada:

```txt
/path/to/stabilityai/stable-diffusion-xl-base-1.0
```

es solo un marcador de posición. Reemplácela por una ruta real de un checkpoint local.

### 4. Ejecuciones de `probe_reasoning` muy lentas

Si configura el modelo de razonamiento para una familia con carga pesada de razonamiento como `o4-mini`, esa etapa puede tardar significativamente más que los modelos de finalización de chat normales.

## Archivos que puede querer editar

- [utils/api_client.py](utils/api_client.py): valores predeterminados compartidos del agente y enrutamiento del proveedor
- [.env.example](.env.example): plantilla de entorno del proveedor
- [pipeline_runner.py](pipeline_runner.py): punto de entrada del pipeline de texto
- [vision_pipeline_runner.py](vision_pipeline_runner.py): punto de entrada del pipeline de visión

## Ejemplos Mínimos Funcionando

Pipeline de texto:

```bash
python pipeline_runner.py -i tasks.txt
```

Pipeline de visión con modo de tipografía:

```bash
python vision_pipeline_runner.py -i tasks.txt
```

Pipeline de visión con analizador enrutado por proxy:

```bash
python vision_pipeline_runner.py \
  -i tasks.txt \
  --vision-model gpt-4o-mini \
  --vision-api-env-key PROXY_API_KEY \
  --vision-base-url https://your-proxy.example.com/v1
```
