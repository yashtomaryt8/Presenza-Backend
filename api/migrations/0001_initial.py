from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name='UserProfile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('student_id', models.CharField(blank=True, default='', max_length=50)),
                ('department', models.CharField(blank=True, default='', max_length=100)),
                ('embeddings_json', models.TextField(default='[]')),
                ('photo_count', models.IntegerField(default=0)),
                ('is_present', models.BooleanField(default=False)),
                ('last_seen', models.DateTimeField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='AttendanceLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('event_type', models.CharField(choices=[('entry', 'Entry'), ('exit', 'Exit')], default='entry', max_length=10)),
                ('confidence', models.FloatField(default=0.0)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='logs', to='api.userprofile')),
            ],
            options={'ordering': ['-timestamp']},
        ),
        migrations.CreateModel(
            name='AttendanceSession',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('entry_time', models.DateTimeField()),
                ('exit_time', models.DateTimeField(blank=True, null=True)),
                ('duration_minutes', models.FloatField(blank=True, null=True)),
                ('date', models.DateField()),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='sessions', to='api.userprofile')),
            ],
            options={'ordering': ['-entry_time']},
        ),
    ]
