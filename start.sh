result=$(python -c 'from codebase_reference_rag import CodeBaseReference; print(CodeBaseReference.returnReference("'$1'"))')
echo $result
source new_env.sh
echo "Python environment setup completed."
pip install -r requirements.txt
python git_pilot.py $result