# torchrun --nproc_per_node 8 lib/train/run_training.py --script flextrackv2 --config flextrackv2_b224 --save_dir .
cd RGBE_workspace/results/VisEvent_miss
zip -r flextrackv2_b224_8_E_miss.zip flextrackv2_b224_8/
cd ../../../
mv RGBE_workspace/results/VisEvent_miss/flextrackv2_b224_8_E_miss.zip zip_files/



cd RGBE_workspace/results/VisEvent
zip -r flextrackv2_b224_8_E.zip flextrackv2_b224_8/
cd ../../../
mv RGBE_workspace/results/VisEvent/flextrackv2_b224_8_E.zip zip_files/






cd RGBT_workspace/results/LasHeR_miss/
zip -r flextrackv2_b224_8_T_miss.zip flextrackv2_b224_8/
cd ../../../
mv RGBT_workspace/results/LasHeR_miss/flextrackv2_b224_8_T_miss.zip zip_files/


cd RGBT_workspace/results/LasHeR/
zip -r flextrackv2_b224_8_T.zip flextrackv2_b224_8/
cd ../../../
mv RGBT_workspace/results/LasHeR/flextrackv2_b224_8_T.zip zip_files/

cd RGBT_workspace/results/RGBT234/
zip -r flextrackv2_b224_8_T_234.zip flextrackv2_b224_8/
cd ../../../
mv RGBT_workspace/results/RGBT234/flextrackv2_b224_8_T_234.zip zip_files/


cd RGBT_workspace/results/RGBT234_miss/
zip -r flextrackv2_b224_8_T_234_miss.zip flextrackv2_b224_8/
cd ../../../
mv RGBT_workspace/results/RGBT234_miss/flextrackv2_b224_8_T_234_miss.zip zip_files/