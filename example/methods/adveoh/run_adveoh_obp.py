import sys

sys.path.append('../../../')  # This is for finding all the modules

from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.adveoh import AdvEoH
from llm4ad.method.adveoh.profiler import AdvEoHProfiler
from llm4ad.method.adveoh.adapters.obp import AdvOBPEvaluation, OBPInstanceGenEvaluation


def main():
    # --- Heuristic LLM (e.g. Gemini for reasoning) ---
    llm_heu = HttpsApi(host='api.bltcy.ai',
                       key='sk-xxx',  # your key
                       model='gemini-2.5-flash',
                       timeout=120)

    # --- Instance LLM (can use a different model/key) ---
    llm_inst = HttpsApi(host='api.bltcy.ai',
                        key='sk-xxx',  # your key
                        model='gpt-4o-mini',
                        timeout=120)

    # --- Task-specific evaluations (from the adapter) ---
    # AdvOBPEvaluation subclasses OBPEvaluation and accepts `instances=None` kwarg
    # OBPInstanceGenEvaluation validates generate_instance(seed) output
    heu_evaluation = AdvOBPEvaluation()
    inst_evaluation = OBPInstanceGenEvaluation()

    # --- Method ---
    method = AdvEoH(
        llm_heu=llm_heu,
        llm_inst=llm_inst,
        heu_evaluation=heu_evaluation,
        inst_evaluation=inst_evaluation,
        profiler=AdvEoHProfiler(log_dir='logs/adveoh_obp', log_style='simple'),
        max_generations=10,
        pop_size_heu=10,
        pop_size_inst=10,
        samples_per_gen_heu=8,
        samples_per_gen_inst=4,
        n_inst_sample=5,
        n_seed=3,
        n_opponent_desc=3,
        num_evaluators=4,
        debug_mode=False,
    )

    method.run()


if __name__ == '__main__':
    main()
