from app.storage.database import get_supabase

client = get_supabase()

preds = client.table("ml_predictions").select("*").execute().data
comps = client.table("components").select("*").execute().data

print("=== ml_predictions ===")
for p in preds:
    print(f"  {p['dut_id']} | {p['anomaly_class']} | score={p['anomaly_score']} | {p['overall_verdict']}")

print("\n=== components ===")
for c in comps:
    print(f"  {c['dut_id']} | {c['part_type']} | lot={c['lot_number']}")
