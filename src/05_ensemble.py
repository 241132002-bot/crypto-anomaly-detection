import numpy as np, json, os
from sklearn.metrics import (classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score, accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('results', exist_ok=True)
y_test    = np.load('data/processed/y_test.npy')
if_scores = np.load('results/if_scores.npy')
rf_scores = np.load('results/rf_scores.npy')
if_pred   = np.load('results/if_predictions.npy')
rf_pred   = np.load('results/rf_predictions.npy')

gs_available = False
if os.path.exists('results/gs_scores.npy'):
    gs_s = np.load('results/gs_scores.npy')
    if gs_s.shape == if_scores.shape:
        gs_available = True
        gs_scores = gs_s
        gs_pred = np.load('results/gs_predictions.npy')

ens_scores = (0.20*if_scores + 0.50*rf_scores + 0.30*gs_scores) if gs_available else (0.30*if_scores + 0.70*rf_scores)
ens_pred   = (ens_scores >= 0.5).astype(int)

models = {'Isolation Forest':(if_pred,if_scores),'Random Forest':(rf_pred,rf_scores),'Ансамблов модел':(ens_pred,ens_scores)}
if gs_available:
    models['GraphSAGE'] = (gs_pred, gs_scores)

print("="*60)
print(f"{'Модел':<20} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
print("-"*60)
results = {}
colors  = ['steelblue','forestgreen','darkorange','crimson']
for name,(pred,score) in models.items():
    acc=accuracy_score(y_test,pred); prec=precision_score(y_test,pred,zero_division=0)
    rec=recall_score(y_test,pred,zero_division=0); f1=f1_score(y_test,pred,zero_division=0)
    auc=roc_auc_score(y_test,score)
    results[name]=dict(Accuracy=acc,Precision=prec,Recall=rec,F1=f1,AUC=auc)
    print(f"{name:<20} {acc:>6.3f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {auc:>6.3f}")
print("-"*60)

cm = confusion_matrix(y_test,ens_pred)
print(f"\nConfusion Matrix — Ансамблов:\n  TN={cm[0,0]} FP={cm[0,1]}\n  FN={cm[1,0]} TP={cm[1,1]}")

fig,axes=plt.subplots(2,2,figsize=(14,10))
fig.suptitle('Сравнение — Реални данни',fontsize=14,fontweight='bold')
clrs=colors[:len(models)]; lws=[1.5]*(len(models)-1)+[2.5]

ax=axes[0,0]
for (name,(pred,score)),c,lw in zip(models.items(),clrs,lws):
    fpr,tpr,_=roc_curve(y_test,score)
    ax.plot(fpr,tpr,color=c,lw=lw,label=f'{name} ({results[name]["AUC"]:.3f})')
ax.plot([0,1],[0,1],'k--'); ax.set_title('ROC криви'); ax.legend(fontsize=8)

ax=axes[0,1]
ml=['Accuracy','Precision','Recall','F1','AUC']; x=np.arange(5); w=0.8/len(models)
for i,(name,vals) in enumerate(results.items()):
    ax.bar(x+i*w,[vals[m] for m in ml],w,label=name,color=clrs[i],alpha=0.85)
ax.set_xticks(x+w*(len(models)-1)/2); ax.set_xticklabels(ml); ax.set_ylim(0,1.15)
ax.set_title('Метрики'); ax.legend(fontsize=7)

sns.heatmap(cm,annot=True,fmt='d',cmap='Reds',ax=axes[1,0],
    xticklabels=['Нормална','Аномалия'],yticklabels=['Нормална','Аномалия'])
axes[1,0].set_title('Confusion Matrix — Ансамбъл')

ax=axes[1,1]
for (name,(pred,score)),c,lw in zip(models.items(),clrs,lws):
    p,r,_=precision_recall_curve(y_test,score)
    ax.plot(r,p,color=c,lw=lw,label=name)
ax.set_title('Precision-Recall'); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('results/04_ensemble_comparison.png',dpi=150,bbox_inches='tight')
plt.show()

with open('results/final_results.json','w',encoding='utf-8') as f:
    json.dump({k:{m:round(v,4) for m,v in vals.items()} for k,vals in results.items()},f,ensure_ascii=False,indent=2)
np.save('results/ensemble_scores.npy',ens_scores)
np.save('results/ensemble_predictions.npy',ens_pred)
print("\nГотово! results/final_results.json")
