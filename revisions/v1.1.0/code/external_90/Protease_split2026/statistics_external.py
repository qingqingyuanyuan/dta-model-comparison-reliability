"""Conditional entity resampling, rank migration and chemical-space descriptors."""
import itertools
import numpy as np
import pandas as pd
from functools import partial
pd_read_csv = partial(pd.read_csv, float_precision='round_trip')
from scipy.stats import kendalltau, rankdata
from common_external import ROOT, MODELS, SETTINGS, SEEDS, run_id, now, read_json, write_json, verify_freeze
from metrics_original import concordance_index

def weighted_ci(y,p,w):
    """O(n log n) C-index for integer cluster multiplicities; excludes label ties."""
    y=np.asarray(y);p=np.asarray(p);w=np.asarray(w,dtype=np.int64)
    keep=w>0;y=y[keep];p=p[keep];w=w[keep]
    if len(y)<2 or len(np.unique(y))<2:return float('nan')
    _,r=np.unique(p,return_inverse=True);r=r+1
    order=np.argsort(y,kind='mergesort');y=y[order];r=r[order];w=w[order]
    tree=np.zeros(int(r.max())+1,dtype=np.int64)
    def prefix(i):
        total=0
        while i>0:total+=int(tree[i]);i-=i&-i
        return total
    previous=0;denominator=0;numerator=0.;start=0
    while start<len(y):
        end=start+1
        while end<len(y) and y[end]==y[start]:end+=1
        for i in range(start,end):
            less=prefix(int(r[i])-1);equal=prefix(int(r[i]))-less
            numerator+=int(w[i])*(less+.5*equal);denominator+=int(w[i])*previous
        for i in range(start,end):
            j=int(r[i])
            while j<len(tree):tree[j]+=w[i];j+=j&-j
            previous+=int(w[i])
        start=end
    return numerator/denominator if denominator else float('nan')

def tau(a,b):
    r=kendalltau(a,b,method='auto')
    return {'tau':float(r.statistic),'p_two_sided':float(r.pvalue)}

def main():
    verify_freeze();p=read_json(ROOT/'protocol.json');out=ROOT/'results/statistics';out.mkdir(exist_ok=True)
    metrics=pd_read_csv(ROOT/'results/ALL_30_METRICS.csv')
    if len(metrics)!=30:raise ValueError('Need all 30 evaluation rows')
    summary=metrics.groupby(['setting','model']).agg(C_index_mean=('C_index','mean'),C_index_SD=('C_index',lambda x:float(np.std(x,ddof=1))),
                 RMSE_pKi_mean=('RMSE_pKi','mean'),RMSE_pKi_SD=('RMSE_pKi',lambda x:float(np.std(x,ddof=1))),
                 MAE_pKi_mean=('MAE_pKi','mean'),unique_C_index_mean=('unique_C_index','mean')).reset_index()
    summary['rank']=summary.groupby('setting').C_index_mean.rank(ascending=False,method='average')
    summary.to_csv(out/'MEAN_SD_AND_RANKS.csv',index=False)
    means={s:summary[summary.setting==s].set_index('model').loc[MODELS].C_index_mean.to_numpy() for s in SETTINGS}
    correlations=[{'comparison':'cold_drug_vs_cold_target','aggregation':'mean_of_per_seed_record_C_index',**tau(means[SETTINGS[0]],means[SETTINGS[1]])}]
    for seed in SEEDS:
        scores=[metrics[(metrics.setting==s)&(metrics.seed==seed)].set_index('model').loc[MODELS].C_index.to_numpy() for s in SETTINGS]
        correlations.append({'comparison':f'cold_drug_vs_cold_target_seed{seed}','aggregation':'per_seed_record_C_index',**tau(*scores)})
    pd.DataFrame(correlations).to_csv(out/'CROSS_REGIME_RANK_CORRELATIONS.csv',index=False)
    rankrows=[];pairrows=[];macrorows=[]
    for si,setting in enumerate(SETTINGS):
        data=[];reference=None
        for name in MODELS:
            seeds=[]
            for seed in SEEDS:
                f=pd_read_csv(ROOT/'results/runs'/run_id(setting,name,seed)/'test_unique_input_predictions.csv')
                f=f.sort_values(['Canonical SMILES','Target Sequence']).reset_index(drop=True)
                identity=f[['Canonical SMILES','Target Sequence','Label']]
                if reference is None:reference=identity
                elif not reference.equals(identity):raise ValueError('Prediction alignment mismatch')
                seeds.append(f.Prediction_pKi.to_numpy())
            data.append(seeds)
        predictions=np.asarray(data);y=reference.Label.to_numpy()
        key='Canonical SMILES' if setting=='cold_drug' else 'Target Sequence'
        entities,codes=np.unique(reference[key],return_inverse=True);n=len(entities)
        original=np.array([[concordance_index(y,predictions[m,k]) for k in range(3)] for m in range(5)]).mean(axis=1)
        rng=np.random.default_rng(p['bootstrap_seed']+si);boot=[]
        for b in range(p['bootstrap_replicates']):
            multiplicity=rng.multinomial(n,np.full(n,1/n));weights=multiplicity[codes]
            values=np.array([[weighted_ci(y,predictions[m,k],weights) for k in range(3)] for m in range(5)]).mean(axis=1)
            if np.isfinite(values).all():boot.append(values)
            if (b+1)%50==0:print('BOOTSTRAP',setting,b+1,'/',p['bootstrap_replicates'],flush=True)
        boot=np.asarray(boot)
        if len(boot)<.9*p['bootstrap_replicates']:raise ValueError('Too many undefined bootstrap C-indices')
        pd.DataFrame(boot,columns=MODELS).to_csv(out/f'{setting}_BOOTSTRAP_CI.csv',index=False)
        ranks=np.array([rankdata(-row,method='average') for row in boot])
        for mi,name in enumerate(MODELS):
            rankrows.append({'setting':setting,'model':name,'entity_count':n,'valid_resamples':len(boot),
                             'unique_C_index_mean':original[mi],'rank_low_2.5pct':float(np.quantile(ranks[:,mi],.025)),
                             'rank_high_97.5pct':float(np.quantile(ranks[:,mi],.975)),'P_rank1':float(np.mean(ranks[:,mi]==1))})
            valid=0;per=[]
            for e in range(n):
                ix=codes==e
                if np.unique(y[ix]).size<2:continue
                valid+=1;per.append(np.mean([concordance_index(y[ix],predictions[mi,k,ix]) for k in range(3)]))
            macrorows.append({'setting':setting,'model':name,'valid_entities':valid,'macro_mean_per_entity_per_seed_CI':float(np.mean(per)) if per else None})
        for a,b in itertools.combinations(range(5),2):
            delta=boot[:,a]-boot[:,b]
            pairrows.append({'setting':setting,'model_A':MODELS[a],'model_B':MODELS[b],
                             'mean_unique_CI_A_minus_B':float(original[a]-original[b]),
                             'conditional_percentile_low':float(np.quantile(delta,.025)),
                             'conditional_percentile_high':float(np.quantile(delta,.975)),
                             'fraction_A_gt_B':float(np.mean(delta>0))})
    pd.DataFrame(rankrows).to_csv(out/'CONDITIONAL_RANK_INTERVALS.csv',index=False)
    pd.DataFrame(pairrows).to_csv(out/'CONDITIONAL_PAIRED_CONTRASTS.csv',index=False)
    pd.DataFrame(macrorows).to_csv(out/'MACRO_PER_ENTITY.csv',index=False)
    chemical(out)
    report='''# 非激酶外部验证完成\n\n30 组 = 5 个原始配置 × 2 个冷启动场景 × 3 个种子。\n\n这是在人类蛋白酶 Ki 队列内从头训练的扩展评估，不是激酶模型的零样本迁移。\n\n- MEAN_SD_AND_RANKS.csv：逐种子 C-index 的均值、样本 SD 和排名。\n- CROSS_REGIME_RANK_CORRELATIONS.csv：均值排名及逐种子排名相关。五模型相关的 p 值不等同于训练过程总体显著性。\n- CONDITIONAL_RANK_INTERVALS.csv / CONDITIONAL_PAIRED_CONTRASTS.csv：固定划分和三个已训练种子条件下的实体 bootstrap，不能替代独立划分或更多训练种子。\n- MACRO_PER_ENTITY.csv：按药物/蛋白分别计算后平均，与 pooled C-index 分开解释。\n- CHEMICAL_SIMILARITY.csv / SIMILARITY_BIN_METRICS.csv：ECFP4 与 Murcko 骨架描述及分层性能。\n\n本文档对应同一蛋白酶队列的一个冻结划分；两套划分属于同队列敏感性评估，不能视作两个独立外部队列；cold-target 按完整序列隔离，不代表低序列相似度隔离。测试化合物仍可能与训练集相似。数据仅采用精确正值 Ki，排除了带不等号的删失值；不同实验条件仍可能带来异质性。保留原始五模型参数量差异，不主张融合机制因果优势。\n\n请据实际结果修改 Reviewer 3 Comment 5、正文方法/结果/局限和数据可用性，不能直接声称已证明所有蛋白家族泛化。\n'''
    (out/'READ_RESULTS_中文.md').write_text(report,encoding='utf-8')
    write_json(ROOT/'results/ALL_DONE.json',{'completed_utc':now(),'training_runs':30,'statistics_complete':True})
    print('ALL 30 RUNS AND STATISTICS COMPLETE',flush=True)

def chemical(out):
    from rdkit import Chem,DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    folder=ROOT/'prepared/cold_drug'
    train=pd_read_csv(folder/'train.csv');test=pd_read_csv(folder/'test.csv')
    train_smiles=sorted(train['Canonical SMILES'].unique())
    mols=[Chem.MolFromSmiles(s) for s in train_smiles];fps=[gen.GetFingerprint(m) for m in mols]
    scaffolds={MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols};scaffolds.discard('')
    records=[]
    for s in sorted(test['Canonical SMILES'].unique()):
        mol=Chem.MolFromSmiles(s);scaf=MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        sim=max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(mol),fps))
        records.append({'Canonical SMILES':s,'max_train_ECFP4_Tanimoto':sim,'Murcko_scaffold':scaf,
                        'nonempty_scaffold_shared':bool(scaf and scaf in scaffolds)})
    sims=pd.DataFrame(records);sims.to_csv(out/'CHEMICAL_SIMILARITY.csv',index=False)
    rows=[]
    for name in MODELS:
        for seed in SEEDS:
            f=pd_read_csv(ROOT/'results/runs'/run_id('cold_drug',name,seed)/'test_unique_input_predictions.csv').merge(sims,on='Canonical SMILES',validate='many_to_one')
            for lo,hi in [(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.0000001)]:
                sub=f[(f.max_train_ECFP4_Tanimoto>=lo)&(f.max_train_ECFP4_Tanimoto<hi)]
                score=float(concordance_index(sub.Label.to_numpy(),sub.Prediction_pKi.to_numpy())) if len(sub)>1 and sub.Label.nunique()>1 else None
                rows.append({'model':name,'seed':seed,'lower':lo,'upper':min(hi,1.),'records':len(sub),
                             'unique_drugs':sub['Canonical SMILES'].nunique(),'C_index':score})
    pd.DataFrame(rows).to_csv(out/'SIMILARITY_BIN_METRICS.csv',index=False)

if __name__=='__main__':main()
