echo "2b - 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "1a-8-hidden) iql_zeta_0.2_rnd_PointmassEasy-v0"
echo " "
echo "python run_iql.py --env_name PointmassEasy-v0 \
--exp_name iql_zeta_0.2_rnd --use_rnd \
--num_exploration_steps=20000 \
--unsupervised_exploration \
--awac_lambda=1 \
--iql_expectile=0.2"

python run_iql.py --env_name PointmassEasy-v0 \
--exp_name iql_zeta_0.2_rnd --use_rnd \
--num_exploration_steps=20000 \
--unsupervised_exploration \
--awac_lambda=1 \
--iql_expectile=0.2


echo "2b - 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "1a-9-hidden) iql_zeta_0.9_rnd_PointmassEasy-v0"
echo " "
echo "python run_iql.py --env_name PointmassEasy-v0 \
--exp_name iql_zeta_0.9_rnd --use_rnd \
--num_exploration_steps=20000 \
--unsupervised_exploration \
--awac_lambda=1 \
--iql_expectile=0.9"
python run_iql.py --env_name PointmassEasy-v0 \
--exp_name iql_zeta_0.9_rnd --use_rnd \
--num_exploration_steps=20000 \
--unsupervised_exploration \
--awac_lambda=1 \
--iql_expectile=0.9


echo "2c - 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "1a-10-hidden) iql_zeta_0.9_rnd_PointmassMedium-v0"
echo " "
echo "python run_iql.py --env_name PointmassMedium-v0 \
--exp_name iql_zeta_0.9_rnd --use_rnd \
--num_exploration_steps=20000 \
--unsupervised_exploration \
--awac_lambda=1 \
--iql_expectile=0.9"    
python run_iql.py --env_name PointmassMedium-v0 \
--exp_name iql_zeta_0.9_rnd --use_rnd \
--num_exploration_steps=20000 \
--unsupervised_exploration \
--awac_lambda=1 \
--iql_expectile=0.9


echo "2c - 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "1a-11-hidden) iql_zeta_0.9_random_PointmassMedium-v0"
echo " "
echo "--exp_name iql_zeta_0.9_random  \
--num_exploration_steps=20000 \
--unsupervised_exploration \
--awac_lambda=1 \
--iql_expectile=0.9"
python run_iql.py --env_name PointmassMedium-v0 \
--exp_name iql_zeta_0.9_random  \
--num_exploration_steps=20000 \
--unsupervised_exploration \
--awac_lambda=1 \
--iql_expectile=0.9

#3b
echo "3b - 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "2a-3-hidden) cql_alpha_0.0_rnd_PointmassHard-v0"
echo " "
echo "python run_cql.py --env_name PointmassHard-v0 \
--exp_name cql_alpha_0.0_rnd \
--use_rnd --unsupervised_exploration \
--offline_exploitation --cql_alpha=0.0"
python run_cql.py --env_name PointmassHard-v0 \
--exp_name cql_alpha_0.0_rnd \
--use_rnd --unsupervised_exploration \
--offline_exploitation --cql_alpha=0.0

echo "3b - 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "2a-4-hidden) cql_alpha_0.1_rnd_PointmassHard-v0"
echo " "
echo "python run_cql.py --env_name PointmassHard-v0 \
--exp_name cql_alpha_0.1_rnd \
--use_rnd --unsupervised_exploration \
--offline_exploitation --cql_alpha=0.1"
python run_cql.py --env_name PointmassHard-v0 \
--exp_name cql_alpha_0.1_rnd \
--use_rnd --unsupervised_exploration \
--offline_exploitation --cql_alpha=0.1

#3c May need to run 3 times
echo "3c - 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "2a-5-hidden) cql_alpha_0.1_random_PointmassHard-v0"
echo " "
echo "python run_cql.py --env_name PointmassHard-v0 \
--exp_name cql_alpha_0.1_random \
--unsupervised_exploration \
--offline_exploitation --cql_alpha=0.1"
python run_cql.py --env_name PointmassHard-v0 \
--exp_name cql_alpha_0.1_random \
--unsupervised_exploration \
--offline_exploitation --cql_alpha=0.1








